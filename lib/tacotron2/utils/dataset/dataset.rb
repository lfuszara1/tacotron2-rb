require './lib/tacotron2/utils/audio/audio'
require './lib/tacotron2/text/cleaners'

class Dataset

  def initialize
    @config = YAML.safe_load(File.read('lib/tacotron2/config.yml'), symbolize_names: true)
    @audio = Audio.new
    @text = Cleaners.new
  end

  def files_to_list(file_dir)
    file_list = []
    File.foreach(File.join(file_dir, 'metadata.csv')) do |line|
      parts = line.strip.split('|')
      wav_path = File.join(file_dir, 'wavs', parts[0] + '.wav')
      if @config[:prep]
        file_list << [get_mel_text_pair(parts[1], wav_path),wav_path]
      else
        file_list << [parts[1], wav_path]
      end
    end
    if @config[:prep] != nil && @config[:pth] != nil
      File.open(File.join(@config[:pth]), 'wb') do |file|
        file.write(Marshal.dump(file_list))
      end
    end
  end

  def get_mel_text_pair(text, wav_path)
    text = get_text(text)
    mel = get_mel(wav_path)
    [text, mel]
  end

  def get_text(text)
    Torch.tensor(@text.text_to_sequence(text, @config[:text_cleaners]), dtype: :int16)
  end

  def get_mel(wav_path)
    wav = @audio.load_audio(wav_path)
    Torch.tensor(@audio.melspectrogram(wav), dtype: :float32)
  end

end

ds = Dataset.new
ds.files_to_list(File.join('data', 'LJSpeech-1.1'))
