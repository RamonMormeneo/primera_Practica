#pytube

# importamos la librería

from pytube import YouTube

#si ocurre el error 403

from pytube.innertube import _default_clients

_default_clients['ANDROID']['context']['client']['clientVersion'] = '19.08.35'
_default_clients['IOS']['context']['client']['clientVersion'] = '19.08.35'
_default_clients['ANDROID_EMBED']['context']['client']['clientVersion'] = '19.08.35'
_default_clients['IOS_EMBED']['context']['client']['clientVersion'] = '19.08.35'
_default_clients['IOS_MUSIC']['context']['client']['clientVersion'] = '6.41'
_default_clients['ANDROID_MUSIC'] = _default_clients['ANDROID']

# url del video de YouTube (Casiopea - Looking Up/Dr. Solo/Bass Solo *Live 1985*  -  8:29)

url = 'https://www.youtube.com/watch?v=S0Xm1PWb07o'

# creamos el objeto YouTube, la conexión al video

youtube = YouTube(url)

# titulo del video

youtube.title

# duración del video, en minutos

round(youtube.length / 60, 2)

# descripción del video

print(youtube.description)

# lista de formatos disponibles

for stream in youtube.streams:
    print(stream)

# seleccionamos la mejor calidad de video disponible

video = youtube.streams.get_highest_resolution()

# nombre por defecto del archivo

video.default_filename


# descargamos el video

ruta = '../../../files'

video.download(output_path=ruta)

# todos los formatos de audio

for stream in youtube.streams.filter(only_audio=True):
    print(stream)

# de entre todos los formatos de audio, seleccionamos el primero

audio = youtube.streams.filter(only_audio=True).first()

# descargamos el audio

audio.download(output_path=ruta)

# url (How AI Could Empower Any Business | Andrew Ng | TED  -  11:16)

url_ted = 'https://www.youtube.com/watch?v=reUZRyXxUs4'

# creamos el objeto YouTube, la conexión al video, y seleccionamos el primero

youtube = YouTube(url_ted)

video = youtube.streams.first()

# todos los subtitulos disponibles

youtube.captions

# descargar subtítulos en español

subs = youtube.captions['es']

# conversión a formato SRT

texto_srt = subs.generate_srt_captions()

texto_srt[:1000]

# ruta guardado subtítulos 

ruta_srt = ruta + '/' + youtube.title + '.srt'

ruta_srt

# guardar subtítulos en un archivo

with open(ruta_srt, 'w') as f:
    
    f.write(texto_srt)

# importamos objeto Playlist

from pytube import Playlist

# url de la playlist

play_url = 'https://www.youtube.com/watch?v=vegqkKGTsBc&list=PL8AC3254DFDA8A81B'

# creamos el objeto Playlist

playlist = Playlist(play_url)

# titulo de la playlist

playlist.title

# ruta de guardado

ruta_play = ruta + '/' + playlist.title


# descargamos todos los videos de la playlist

for video in playlist.videos:
    
    video.streams.get_highest_resolution().download(output_path=ruta_play)
    
    print(f'Descargado: {video.title}')

# podemos descargar el audio de la playlist

for url in playlist:
    
    youtube = YouTube(url)
    
    audio = youtube.streams.filter(only_audio=True).first()
    
    audio.download(output_path=ruta_play)
    
    print(f'Descargado: {youtube.title}')
