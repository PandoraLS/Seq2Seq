import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from trainer.base_trainer import BaseTrainer
from util.utils import set_requires_grad, compute_STOI, compute_PESQ

plt.switch_backend("agg")


class Trainer(BaseTrainer):
    def __init__(self,
                 config: dict,
                 resume: bool,
                 generator,
                 discriminator,
                 generator_optimizer,
                 discriminator_optimizer,
                 additional_loss_function,
                 train_dl,
                 validation_dl):
        super().__init__(config, resume, generator, discriminator, generator_optimizer, discriminator_optimizer, additional_loss_function)
        self.train_dataloader = train_dl
        self.validation_dataloader = validation_dl

    def _visualize_weights_and_grads(self, model, epoch):
        for name, param in model.named_parameters():
            self.writer.add_histogram("WEIGHT_" + name, param.clone().cpu().data.numpy(), epoch)
            self.writer.add_histogram("GRAD_" + name, param.grad.cpu().numpy(), epoch)

    def _train_epoch(self, epoch):
        for i, (noisy, clean, name) in enumerate(self.train_dataloader, start=1):
            # For visualization
            batch_size = self.train_dataloader.batch_size
            n_batch = len(self.train_dataloader)
            n_iter = n_batch * batch_size * (epoch - 1) + i * batch_size

            noisy = noisy.to(self.device)  # [batch_size, 1, sample_length]
            clean = clean.to(self.device)
            enhanced = self.generator(noisy)

            """================ Optimize D ================"""
            set_requires_grad(self.discriminator, True)
            self.optimizer_D.zero_grad()

            # D(noisy, enhanced) => Fake
            enhanced_conditioned_noisy = torch.cat((noisy, enhanced), dim=1)
            enhanced_in_D = self.discriminator(enhanced_conditioned_noisy.detach())
            enhanced_in_D_loss = self.adversarial_loss_function(enhanced_in_D, torch.zeros(enhanced_in_D.shape).to(self.device))

            # D(noisy, clean) => Real
            clean_conditioned_noisy = torch.cat((noisy, clean), dim=1)
            clean_in_D = self.discriminator(clean_conditioned_noisy)
            real_label = np.random.uniform(0.7, 1.0) if self.soft_label else 1.0
            clean_in_D_loss = self.adversarial_loss_function(clean_in_D, torch.full(clean_in_D.shape, real_label).to(self.device))

            # Combine
            loss_D = (enhanced_in_D_loss + clean_in_D_loss) * 0.5 * self.adversarial_loss_factor
            loss_D.backward()
            self.optimizer_D.step()

            with torch.no_grad():
                self.writer.add_scalars(f"判别器/纯净语音判别结果", {
                    "判别器优化前": torch.mean(clean_in_D),
                    "判别器优化后": torch.mean(self.discriminator(clean_conditioned_noisy)),
                }, n_iter)

                self.writer.add_scalars(f"判别器/纯净语音的判别损失", {
                    "判别器优化前": clean_in_D_loss,
                    "判别器优化后": self.adversarial_loss_function(self.discriminator(clean_conditioned_noisy), torch.full(clean_in_D.shape, real_label).to(self.device))
                }, n_iter)

                self.writer.add_scalars(f"判别器/被增强语音的判别结果", {
                    "判别器优化前": torch.mean(enhanced_in_D),
                    "判别器优化后": torch.mean(self.discriminator(enhanced_conditioned_noisy))
                }, n_iter)

                self.writer.add_scalars(f"判别器/被增强语音的判别损失", {
                    "判别器优化前": enhanced_in_D_loss,
                    "判别器优化后": self.adversarial_loss_function(self.discriminator(enhanced_conditioned_noisy), torch.zeros(enhanced_in_D.shape).to(self.device)),
                }, n_iter)

                self.writer.add_scalar(f"判别器/总损失", loss_D, n_iter)


            """================ Optimize G ================"""
            set_requires_grad(self.discriminator, False)
            self.optimizer_G.zero_grad()

            # D(noisy, enhanced) => Real
            enhanced_conditioned_noisy = torch.cat((noisy, enhanced), dim=1)  # 是否有必要再来这么一次
            enhanced_in_G = self.discriminator(enhanced_conditioned_noisy)
            enhanced_in_G_loss = self.adversarial_loss_function(enhanced_in_G, torch.ones(enhanced_in_G.shape).to(self.device)) * self.adversarial_loss_factor
            additional_loss = self.additional_loss_function(enhanced, clean) * self.additional_loss_factor

            # Combine
            loss_G = enhanced_in_G_loss + additional_loss
            loss_G.backward()
            self.optimizer_G.step()

            with torch.no_grad():
                enhanced = self.generator(noisy)
                enhanced_conditioned_noisy = torch.cat((noisy, enhanced), dim=1)

                self.writer.add_scalars(f"生成器/额外损失", {
                    "生成器优化前": additional_loss,
                    "生成器优化后": self.additional_loss_function(enhanced, clean) * self.additional_loss_factor,
                }, n_iter)

                self.writer.add_scalars(f"生成器/被增强语音的判别结果", {
                    "生成器优化前": torch.mean(enhanced_in_G),
                    "生成器优化后": torch.mean(self.discriminator(enhanced_conditioned_noisy))
                }, n_iter)

                self.writer.add_scalars(f"生成器/被增强语音的判别损失", {
                    "生成器优化前": enhanced_in_G_loss,
                    "生成器优化后": self.adversarial_loss_function(self.discriminator(enhanced_conditioned_noisy), torch.ones(enhanced_in_G.shape).to(self.device)) * self.adversarial_loss_factor
                }, n_iter)

                self.writer.add_scalars(f"生成器/总损失", {
                    "生成器优化前": loss_G,
                    "生成器优化后": self.adversarial_loss_function(self.discriminator(enhanced_conditioned_noisy), torch.ones(enhanced_in_G.shape).to(self.device)) * self.adversarial_loss_factor
                              + self.additional_loss_function(enhanced, clean) * self.additional_loss_factor
                }, n_iter)


    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]
        sample_length = self.validation_custom_config["sample_length"]

        stoi_clean_and_noisy = []
        stoi_clean_and_enhanced = []
        pesq_clean_and_noisy = []
        pesq_clean_and_enhanced = []

        for i, (noisy, clean, name) in enumerate(self.validation_dataloader, start=1):
            assert len(name) == 1, "The batch size of validation dataloader should be 1."
            name = name[0]

            noisy = noisy.to(self.device)
            norm_max = torch.max(noisy).item()
            norm_min = torch.min(noisy).item()
            noisy = 2 * (noisy - norm_min) / (norm_max - norm_min) - 1

            assert noisy.dim() == 3
            noisy_chunks = torch.split(noisy, sample_length, dim=2)
            if noisy_chunks[-1].shape[-1] != sample_length:
                # Delete tail
                noisy_chunks = noisy_chunks[:-1]

            enhanced_chunks = []
            for noisy_chunk in noisy_chunks:
                enhanced_chunks.append(self.generator(noisy_chunk).detach().cpu().numpy().reshape(-1))

            enhanced = np.concatenate(enhanced_chunks)
            enhanced = (enhanced + 1) * (norm_max - norm_min) / 2 + norm_min

            noisy = noisy.cpu().numpy().reshape(-1)[:len(enhanced)]
            clean = clean.cpu().numpy().reshape(-1)[:len(enhanced)]

            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_noisy", noisy, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_clean", clean, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_enhanced", enhanced, epoch, sample_rate=16000)

            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([noisy, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveplot(y, sr=16000, ax=ax[j])
                plt.tight_layout()
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for j, y in enumerate([noisy, clean, enhanced]):
                    mag, _ = librosa.magphase(librosa.stft(y, n_fft=320, hop_length=160, win_length=320))
                    axes[j].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[j], sr=16000, hop_length=160)

                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metrics
            stoi_clean_and_noisy.append(compute_STOI(clean, noisy, sr=16000))
            stoi_clean_and_enhanced.append(compute_STOI(clean, enhanced, sr=16000))
            pesq_clean_and_noisy.append(compute_PESQ(clean, noisy, sr=16000))
            pesq_clean_and_enhanced.append(compute_PESQ(clean, enhanced, sr=16000))

        self.writer.add_scalars(f"Metrics/STOI", {
            "clean and noisy": np.mean(stoi_clean_and_noisy),
            "clean and enhanced": np.mean(stoi_clean_and_enhanced)
        }, epoch)
        self.writer.add_scalars(f"Metrics/PESQ", {
            "clean and noisy": np.mean(pesq_clean_and_noisy),
            "clean and enhanced": np.mean(pesq_clean_and_enhanced)
        }, epoch)

        score = (np.mean(stoi_clean_and_enhanced) + self._transform_pesq_range(np.mean(pesq_clean_and_enhanced))) / 2
        return score
