require 'yaml'

require 'numo/narray'

require 'torch.rb'
require 'torchaudio'

class Audio

  def initialize
    @config = YAML.safe_load(File.read('lib/tacotron2/config.yml'), symbolize_names: true)
    @n_fft = (@config[:num_freq] - 1) * 2
    @hop_length = @config[:frame_shift]
    @win_length = @config[:frame_length]
  end

  def load_audio(file_path)
    waveform, sample_rate = TorchAudio.load(file_path)

    waveform = Torch.tensor(waveform)
    waveform = waveform / waveform.abs.max

    return nil unless sample_rate == @config[:sample_rate]

    waveform
  end

  def save_wav(waveform, file_path)
    TorchAudio.save(file_path, waveform, @config[:sample_rate])
  end

  def preemphasis(x)
    a_coeffs = Torch.tensor([1, 1])
    b_coeffs = Torch.tensor([1, -1 * @config[:preemphasis]])

    TorchAudio::Functional.lfilter(x, a_coeffs, b_coeffs)
  end

  def inv_preemphasis(x)
    a_coeffs = Torch.tensor([1, 1])
    b_coeffs = Torch.tensor([1, -1 * @config[:preemphasis]])

    TorchAudio::Functional.lfilter(x, b_coeffs, a_coeffs)
  end

  def spectrogram(y)
    d = stft(preemphasis(y))
    s = amp_to_db(d.abs) - @config[:ref_level_db]
    normalize(s)
  end

  def inv_spectrogram(spectrogram)
    s = db_to_amp(denormalize(spectrogram) + @config[:ref_level_db])
    inv_preemphasis(griffin_lim(s**@config[:power]))
  end

  private

  def stft(y)
    Torch.stft(y, @n_fft, hop_length: @hop_length, win_length: @win_length, return_complex: false)
  end

  def istft(y)
    Torch.istft(y, @n_fft, hop_length: @hop_length, win_length: @win_length, return_complex: false)
  end

  def griffin_lim(s)
    angles = Numo::NMath.exp(0+2i * Math::PI * Numo::DFloat.new(*s.size).rand).to_a
    s_complex = Numo::DComplex.cast(s)
    y = istft(Torch.from_numo(s_complex * angles))
    @config[:gl_iters].times do |_|
      angles = (0+1i * stft(y).angle).exp
      y = istft(s_complex * angles)
    end
    y
  end

  def linear_to_mel(spectrogram)
    mel_basis = build_mel_basis
    mel_basis.dot spectrogram
  end

  def mel_to_linear(spectrogram)
    mel_basis = build_mel_basis
    inv_mel_basis = mel_basis.pinverse
    inverse = inv_mel_basis.dot spectrogram
    Torch.clamp(inverse, min: 1e-10)
  end

  def build_mel_basis()
    TorchAudio::Transforms::MelSpectrogram.new(sample_rate: @config[:sample_rate], n_fft: @n_fft, n_mels: @config[:num_mels], f_min: @config[:fmin], f_max: @config[:fmax])
  end

  def amp_to_db(x)
    Torch.tensor(20) * Torch.clamp(x, min: 1e-5).log10
  end

  def db_to_amp(x)
    (x * 0.05)**10.0
  end

  def normalize(s)
    Torch.clip((s - @config[:min_level_db]) / (-1.0 * @config[:min_level_db]), 0.0, 1.0)
  end

  def denormalize(s)
    (Torch.clip(s, 0, 1) * (-1.0 * @config[:min_level_db])) + @config[:min_level_db]
  end

end

a = Audio.new
wav = a.load_audio('data/LJ001-0013.wav')
spec =  a.spectrogram(wav)
p a.inv_spectrogram(spec)