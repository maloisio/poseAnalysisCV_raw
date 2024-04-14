from moviepy.editor import VideoFileClip, concatenate_videoclips

video1_path = "videooficial.mp4"
video2_path = video1_path

video1 = VideoFileClip(video1_path)
video2 = VideoFileClip(video2_path)


final_clip = concatenate_videoclips([video1, video2])


final_clip.write_videofile("video_concatenado.mp4")

#por enquanto estou só duplicando mesmo video para testar, futuramente pretendo juntar todos os videos em um só