#imports
!pip install tensorflow_io
import tensorflow as tf
import tensorflow_io as tfio
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython.display import Audio, display
import random
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import librosa
import librosa.display
import soundfile as sf
import IPython.display as ipd
from scipy import stats

#baixando o dataset
data_dir = pathlib.Path('/content/BANCO_BAIXADO')
if not data_dir.exists():
  !kaggle datasets download -d lazyrac00n/speech-activity-detection-datasets -p {data_dir.name} #essa linha é ir no kaggle e pedir "copy API command"
  !unzip {data_dir.name}/speech-activity-detection-datasets.zip -d {data_dir.name}
  !rm {data_dir.name}/speech-activity-detection-datasets.zip

def cria_conjuntos_de_audios(random_audio, file_lists):
  '''
  Função: itera sobre cada diretório de ruído procurando arquivos com o mesmo nome do áudio sorteado.

  Parametros: áudio sorteado, diretórios de ruídos

  Retorna: Uma lista contendo os arquivos de áudio correspondenes ao áudio sorteado.
  (não contém o audio sorteado)
  '''
  conjunto = []
  for file_list in file_lists:
      for file_path in file_list:
          if str(file_path.name)[:4] == str(random_audio.name)[:4]:
              conjunto.append(file_path)
  print(conjunto)
  return conjunto

def reproduz_audio(audio):
  '''
  Reproduz o áudio

  Parametros: arquivo de áudio

  Retorna: nada
  '''
  print(f"Reproduzindo áudio: {str(audio)}")
  display(Audio(filename=str(audio)))

def get_waveform(file_path):
  """
  Lê, decodifica o áudio.

  Parametros:Path do arquivo de audio.

  Retorna: Um tensor da forma de onda.
  """
  audio_binary = tf.io.read_file(file_path)
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  return tf.squeeze(audio, axis=-1)


def get_spectrogram(waveform):
  '''
  Converte waveform em espectrograma

  Parametros: waveform

  Retorna: espectrograma
  '''
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
  waveform = tf.cast(waveform, dtype=tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)

  # Normalização do espectrograma
  spectrogram = spectrogram / tf.reduce_max(spectrogram)

  # Redimensionando o espectrograma para (124, 129, 1)
  spectrogram_resized = tf.image.resize(spectrogram[..., tf.newaxis], (124, 129))  # Adiciona dimensão extra
  return spectrogram_resized

def plot_spectrogram(spectrogram, ax):
  '''
  plota spectrograma

  Parametros: espectrograma e eixo

  Retorna: nada
  '''
  if len(spectrogram.shape) > 2:
      spectrogram = np.squeeze(spectrogram, axis=-1)
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def calculate_snr(original_audio, filtered_audio):
  """
  Calculates the Signal-to-Noise Ratio (SNR) in dB.

  Args:
    original_audio: NumPy ndarray representing the original audio signal.
    filtered_audio: NumPy ndarray representing the filtered audio signal.

  Returns:
    SNR value in dB.
  """
  # Ensure both signals have the same length
  min_len = min(len(original_audio), len(filtered_audio))
  original_audio = original_audio[:min_len]
  filtered_audio = filtered_audio[:min_len]

  # Calculate the power of the original signal
  signal_power = np.sum(original_audio**2)

  # Calculate the power of the noise (difference between original and filtered)
  noise_power = np.sum((original_audio - filtered_audio)**2)

  # Calculate SNR in dB
  snr_db = 10 * np.log10(signal_power / noise_power)
  print(f"SNR: {snr_db} dB")
  return snr_db

def path_to_ndarray(audio_path):
  """Loads an audio file and returns it as a NumPy array.

  Args:
      audio_path: Path to the audio file.

  Returns:
      A NumPy array representing the audio data.
  """
  # Load the audio file
  audio_data, _ = librosa.load(audio_path, sr=None)  # sr=None to keep original sample rate

  # Return the audio data as a NumPy array
  return audio_data

def plot_waveform(waveform, ax, title='Waveform'):
  '''
  plota waveform

  Parametros: waveform e eixo

  Retorna: nada
  '''
  timescale = np.arange(waveform.shape[0])
  ax.plot(timescale, waveform.numpy(), color='blue')
  ax.set_title(title)
  ax.set_xlabel('Samples')
  ax.set_ylabel('Amplitude')
  ax.set_xlim([0, waveform.shape[0]])
  ax.grid()

def compara_waveforms(audio_path1, audio_path2, title='Waveforms Comparison'):
  """
  Plots the waveforms of two audio files on the same graph.

  Args:
    audio_path1: Path to the first audio file.
    audio_path2: Path to the second audio file.
    title: Title of the plot (default: 'Waveforms Comparison').
  """
  # Load the audio files
  audio1, sr1 = librosa.load(audio_path1)
  audio2, sr2 = librosa.load(audio_path2)

  file_name1 = os.path.basename(audio_path1)
  file_name2 = os.path.basename(audio_path2)

  # Trim the longer audio to match the shorter one
  min_len = min(len(audio1), len(audio2))
  audio1 = audio1[:min_len]
  audio2 = audio2[:min_len]

  # Create the time axis for each audio
  time1 = np.arange(0, len(audio1)) / sr1
  time2 = np.arange(0, len(audio2)) / sr2

  # Plot the waveforms
  plt.figure(figsize=(12, 6))
  plt.plot(time1, audio1, label= file_name1)
  plt.plot(time2, audio2, label= file_name2, alpha=0.7)  # alpha for transparency

  plt.title(file_name1 + ' X ' + file_name2)
  plt.xlabel('Time (seconds)')
  plt.ylabel('Amplitude')
  plt.legend()
  plt.grid(True)
  plt.show()


def unet_model(input_shape):
  '''
  cria o modelo U-Net

  Parametros: forma do input

  Retorna: modelo U-Net
  '''
  inputs = layers.Input(shape=input_shape)

  # Encoder
  conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
  conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
  pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

  # Bottleneck
  conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
  conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)

  # Decoder
  up1 = layers.UpSampling2D(size=(2, 2))(conv4)
  resized_up1 = layers.Resizing(conv3.shape[1], conv3.shape[2])(up1)
  concat1 = layers.Concatenate()([resized_up1, conv3])
  conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(concat1)
  conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

  up2 = layers.UpSampling2D(size=(2, 2))(conv5)
  resized_up2 = layers.Resizing(conv2.shape[1], conv2.shape[2])(up2)
  concat2 = layers.Concatenate()([resized_up2, conv2])
  conv6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat2)
  conv6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

  up3 = layers.UpSampling2D(size=(2, 2))(conv6)
  resized_up3 = layers.Resizing(conv1.shape[1], conv1.shape[2])(up3)
  concat3 = layers.Concatenate()([resized_up3, conv1])
  conv7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat3)
  conv7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

  # Saída
  outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv7)

  model = models.Model(inputs=[inputs], outputs=[outputs])
  return model

# Função para criar uma máscara de voz com base no espectrograma
def create_mask(path, sample_rate=16000):
  '''
  cria máscara se a amplitude > threshold
  retorna: espectrograma com uma dimensão, a máscara e sample rate
  '''
  y, sr = librosa.load(path, sr=sample_rate)

  # Ajustando parâmetros para corresponder ao espectrograma da U-Net
  stft = librosa.stft(y, n_fft=255, hop_length=128)
  S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
  '''
  threshold = -20
  mask = np.where(S_db > threshold, 1, 0)
  '''
  threshold = np.mean(S_db) + 0.5 * np.std(S_db)
  mask = np.where(S_db > threshold, 1, 0)


  # Redimensionando a máscara para corresponder à saída da U-Net
  mask_resized = tf.image.resize(mask[..., np.newaxis], (124, 129))  # Redimensiona para (124, 129)
  return mask_resized.numpy().squeeze(), stft, sr  # Remover a dimensão extra

# Criar o dataset
def create_dataset(spectrograms, masks):
  return tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(spectrograms, dtype=tf.float32), masks))

def apply_mask(diretorio, mask):
  '''
  Aplica a máscara binária ao espectrograma e retorna o áudio
  Parametros: diretorio do audio, mascara
  Retorna: audio filtrado
  '''
  y, sr = librosa.load(diretorio, sr=16000)
  S = librosa.stft(y)

  # Ajustar a máscara para que tenha o mesmo tamanho que o espectrograma
  if S.shape != mask.shape:
      mask = np.resize(mask, S.shape)

  masked_S = np.abs(S) * mask
  masked_S_complex = masked_S * np.exp(1j * np.angle(S))
  masked_S_db = librosa.amplitude_to_db(np.abs(masked_S_complex), ref=np.max)
  return masked_S_complex, masked_S_db

def recover_audio(masked_S_complex, diretorio):
  recovered_audio = librosa.istft(masked_S_complex)
  diretorio_str = str(diretorio)
  novo_caminho = diretorio_str.replace(".wav", "_filtrado.wav")
  sf.write(novo_caminho, recovered_audio, sr)
  return novo_caminho

def display_images(display_list, sr):
    plt.figure(figsize=(12, 8))  # Alterado para formato de figura vertical
    title = ['Espectrograma', 'Máscara Binária', 'Espectrograma com máscara aplicada']

    for i in range(len(display_list)):
        librosa.display.specshow(display_list[i], sr=sr, x_axis='time', y_axis='log', cmap='coolwarm')
        plt.title(title[i])
        plt.colorbar(format='%+2.0f dB')
        plt.show()

# Definindo os diretórios para os áudios sem ruído e com ruído
path = pathlib.Path('/content/BANCO_BAIXADO/Data/Audio/Noizeus/NoNoise')
file_nonoise = [file for file in path.iterdir() if file.is_file()]

# Importando áudios dos diretórios ruidosos
file_lists = []
for noise_type in ['Babble', 'Car', 'Restaurant', 'Station', 'Street', 'Train']:
    path_noise = pathlib.Path(f'/content/BANCO_BAIXADO/Data/Audio/Noizeus/{noise_type}')
    file_list = [file for file in path_noise.iterdir() if file.is_file()]
    file_lists.append(file_list)

# Configuração de semente para reprodutibilidade
seed = None
random.seed(seed)

# Obtendo índices únicos aleatórios
num_aleatorio = random.sample(range(len(file_nonoise)), 3)

# Garantindo que conjunto_de_treinamento, conjunto_de_teste e conjunto_de_validacao sejam diretórios
conjunto_de_treinamento = cria_conjuntos_de_audios(file_nonoise[num_aleatorio[0]], file_lists)
conjunto_de_teste = cria_conjuntos_de_audios(file_nonoise[num_aleatorio[1]], file_lists)
conjunto_de_validacao = cria_conjuntos_de_audios(file_nonoise[num_aleatorio[2]], file_lists)

# Plotar espectrogramas
print('\n Áudio com forma de onda e espectrograma correspondentes\n')

# Processar e exibir áudios dos conjuntos, gerando spectrograma e forma de onda
espectrogramas_treinamento = []

for i, audio_file in enumerate(conjunto_de_treinamento):
    #process_audio_pairs(audio_file) #apenas para o widget de audio, não retornando nada
    waveform = get_waveform(str(audio_file))
    espectrogram = get_spectrogram(waveform)
    espectrogramas_treinamento.append(espectrogram)

    #plotando para visualização
    fig, axes = plt.subplots(2, figsize=(12, 8))
    plot_waveform(waveform, axes[0], title=f'Waveform {i + 1} do Conjunto de Treinamento')
    plot_spectrogram(espectrogram.numpy(), axes[1])
    axes[1].set_title(f'Espectrograma {i + 1} do Conjunto de Treinamento')
    plt.subplots_adjust(hspace=0.5) #ajuste para overlap dos gráficos
    plt.show()

espectrogramas_validacao = []
for i, audio_file in enumerate(conjunto_de_validacao):
    waveform = get_waveform(str(audio_file))
    espectrogram = get_spectrogram(waveform)
    espectrogramas_validacao.append(espectrogram)

espectrogramas_teste = []
for i, audio_file in enumerate(conjunto_de_teste):
    waveform = get_waveform(str(audio_file))
    espectrogram = get_spectrogram(waveform)
    espectrogramas_teste.append(espectrogram)

mask_dataset = []
mask1_dataset = []
for file in conjunto_de_treinamento:
    mask, S, sr = create_mask(file)
    mask_dataset.append(mask)

for file in conjunto_de_validacao:
    mask, S, sr = create_mask(file)
    mask1_dataset.append(mask)

masked_S_complex,masked_S_db = apply_mask(conjunto_de_treinamento[0], mask_dataset[0])
audio_filtrado = recover_audio(masked_S_complex, conjunto_de_treinamento[0])


display_images([S, mask, masked_S_db], sr)


print("Reproduzindo áudio original: ", str(conjunto_de_treinamento[0]))
display(Audio(filename=str(conjunto_de_treinamento[0])))

print("Reproduzindo áudio filtrado: ", str(audio_filtrado))
display(Audio(filename=str(audio_filtrado)))

print("Forma dos espectrogramas:", tf.convert_to_tensor(espectrogramas_treinamento).shape)
print("Forma das máscaras:", tf.convert_to_tensor(mask_dataset).shape)

#Modelo U-Net
input_shape = (124, 129, 1)
model = unet_model(input_shape)
model.summary()

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
dataset_treino = create_dataset(espectrogramas_treinamento, mask_dataset)
dataset_treino = dataset_treino.batch(4)

dataset_validacao = create_dataset(espectrogramas_validacao, mask1_dataset)  # Use espectrogramas_validacao aqui
dataset_validacao = dataset_validacao.batch(4)

history = model.fit(dataset_treino, validation_data=dataset_validacao, epochs=30)

#criando um espectrograma de teste: # Supondo que você tenha um espectrograma de teste
# Exemplo: espectrograma_teste deve ter forma (124, 129, 1)

# Adicione uma dimensão extra para o batch
espectrograma_teste = espectrogramas_teste[0][np.newaxis, ...]  # Forma (1, 124, 129, 1)

# Fazer a previsão
mascara_prevista = model.predict(espectrograma_teste)

# A máscara prevista terá forma (1, 124, 129, 1) se você tiver uma saída com uma dimensão adicional
# Remova a dimensão do batch se necessário
mascara_prevista = np.squeeze(mascara_prevista)  # Forma (124, 129, 1) agora

# Visualizar a máscara prevista
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Máscara Prevista")
plt.imshow(mascara_prevista, cmap='gray')  # Ajuste conforme a dimensão da máscara

plt.subplot(1, 2, 2)
plt.title("Espectrograma de Teste")
plt.imshow(espectrograma_teste[0, ..., 0], cmap='gray')  # Ajuste conforme a dimensão do espectrograma

plt.show()

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

apply_mask(conjunto_de_teste[0], mascara_prevista)
# Exemplo de uso
# apply_mask('caminho/para/seu/audio.wav', mascara_prevista)
masked_S_complex1,masked_S_db1 = apply_mask(conjunto_de_teste[0], mascara_prevista)
audio_filtrado1 = recover_audio(masked_S_complex1, conjunto_de_teste[0])
y1, sr1 = librosa.load(conjunto_de_teste[0], sr=16000)
S1 = librosa.stft(y1)
display_images([S1, mascara_prevista, masked_S_db1], sr1)

#widgets de audio
print("Reproduzindo áudio original com ruído: ", str(conjunto_de_teste[0]))
display(Audio(filename=str(conjunto_de_teste[0])))

print("Reproduzindo áudio filtrado: ", str(audio_filtrado1))
display(Audio(filename=str(audio_filtrado1)))
audio_filtrado1 = pathlib.Path(audio_filtrado1)

path = pathlib.Path('/content/BANCO_BAIXADO/Data/Audio/Noizeus/NoNoise')
for file in path.iterdir():
  if str(file.name)[:4] == str(conjunto_de_teste[0].name)[:4]:
    print("Reproduzindo audio original sem ruído: ", str(conjunto_de_teste[0]))
    display(Audio(filename= '/content/BANCO_BAIXADO/Data/Audio/Noizeus/NoNoise/' + str(file.name)))
    audio_original = ('/content/BANCO_BAIXADO/Data/Audio/Noizeus/NoNoise/' + str(file.name))
    break

audio_filtrado1 = pathlib.Path(audio_filtrado1)
audio_ruido = pathlib.Path(conjunto_de_teste[0])
audio_original = pathlib.Path(audio_original)

compara_waveforms(audio_original, audio_filtrado1)
compara_waveforms(audio_original, audio_ruido)
compara_waveforms(audio_ruido, audio_filtrado1)

print (audio_filtrado1)
print (audio_ruido)
print (audio_original)
snr = calculate_snr(path_to_ndarray(audio_original), path_to_ndarray(audio_filtrado1))
snr = calculate_snr(path_to_ndarray(audio_original), path_to_ndarray(audio_ruido))
snr = calculate_snr(path_to_ndarray(audio_ruido), path_to_ndarray(audio_filtrado1))
