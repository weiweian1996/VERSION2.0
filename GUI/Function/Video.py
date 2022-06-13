import av
import cv2
#how to use it.
movie = cv2.VideoCapture('0403.wmv')
print("总帧数是:",movie.get(7))

input = av.open ('0403.wmv')
stream = input.streams.video[0]
# stream.codec_context.skip_frame = 'NONEKEY'
# pts = 1s*FPS(33.2)* TIMEs
s = 1
offset_rate = 33344/s
input.seek(offset= int(0), stream=stream, any_frame= False)

#read the frames
for frame in input.decode(stream):
    print(frame)


#15237864