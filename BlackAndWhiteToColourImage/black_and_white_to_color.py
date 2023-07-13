import numpy as np
import cv2
from cv2 import dnn
import imutils
import os
from os.path import isfile, join
import moviepy.editor
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import glob

fps = 30.0
current_progress = 0.0

#--------Model file paths--------#
proto_file = 'models\colorization_deploy_v2.prototxt'
model_file = 'models\colorization_release_v2.caffemodel'
hull_pts = 'models\pts_in_hull.npy'

#--------Reading the model params--------#
net = dnn.readNetFromCaffe(proto_file,model_file)
kernel = np.load(hull_pts)
#-----------------------------------#---------------------#

# add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
#-----------------------------------#---------------------#

current_progress = 0.0

def get_progress():
    return current_progress


def convert(img, path, img_name):

    #img_path = 'images/' + img_name


    

    #-----Reading and preprocessing image--------#
    #img = cv2.imread(img_path)
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    #-----------------------------------#---------------------#

    

    # we'll resize the image for the network
    resized = cv2.resize(lab_img, (224, 224))
    # split the L channel
    L = cv2.split(resized)[0]
    # mean subtraction
    L -= 50
    #-----------------------------------#---------------------#

    # predicting the ab channels from the input L channel

    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    # resize the predicted 'ab' volume to the same dimensions as our
    # input image
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))


    # Take the L channel from the image
    L = cv2.split(lab_img)[0]
    # Join the L channel with predicted ab channel
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)

    # Then convert the image from Lab to BGR
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # change the image to 0-255 range and convert it from float32 to int
    colorized = (255 * colorized).astype("uint8")

    cv2.imwrite(path + img_name, colorized)


def videoConvert(video_name, width = 500):
    global current_progress
    
    
    vs = cv2.VideoCapture(video_name)

    global fps
    fps = vs.get(cv2.CAP_PROP_FPS)
    frame_count = vs.get(cv2.CAP_PROP_FRAME_COUNT)

    count = 0
    success = True

    while success:#count <= frame_count:
        success, frame = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        convert(frame, "./colored_frames/", "frame%d.jpg" % count)

        count += 1

        progress = count / frame_count * 100
        current_progress = progress * 0.95
        print("%.2f" % progress)
        

    vs.release()
    cv2.destroyAllWindows()


def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)

        #inserting the frames into an image array
        frame_array.append(img)
 
    #out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()



def colorBAWVideo(input_path, pathOut):
    global current_progress
    current_progress = 0.0

    videoConvert(input_path)

    pathIn= "./colored_frames/"
    output= "./stage/video.mp4"

    convert_frames_to_video(pathIn, output, fps)

    

    videoRAW = moviepy.editor.VideoFileClip(input_path)
    audioRAW = videoRAW.audio

    if audioRAW != None:
        audioRAW.write_audiofile("./stage/audio.mp3")

    

        # Open the video and audio
        video_clip = VideoFileClip(output)
        audio_clip = AudioFileClip("./stage/audio.mp3")

        # Concatenate the video clip with the audio clip
        final_clip = video_clip.set_audio(audio_clip)

        # Export the final video with audio
        final_clip.write_videofile(pathOut)
    else:
        video_clip = VideoFileClip(output)
        video_clip.write_videofile(pathOut)


    files = glob.glob("./colored_frames/*")
    for f in files:
        os.remove(f)

    current_progress = 100.0
    





input_video = "./input_video/india.mp4"
output_video = "./output/colored_india_extended.mp4" #path to video + video title

#colorBAWVideo(input_video, output_video)