from gtts import gTTS
from io import BytesIO
from mpg123 import Mpg123, Out123
#import pygame


def main():
    fp = BytesIO()
    tts = gTTS("ててててーん！みかねあさんでしたー！", lang="ja")
    tts.write_to_fp(fp)
    fp.seek(0)

    mp3 = Mpg123()
    mp3.feed(fp.read())

    out = Out123()

    for frame in mp3.iter_frames(out.start):
        out.play(frame)
#    pygame.mixer.init()
#    pygame.mixer.music.load(fp)
#    pygame.mixer.music.play()
#    while pygame.mixer.music.get_busy():
#        pygame.time.Clock().tick(10)

if __name__=='__main__':
    main()
