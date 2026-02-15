import numpy as np
def get_timestamps(times, video_length):

    video_timestamps = np.zeros(video_length)
    video_timestamps[[int(item) for item in times]] = 1

    #Convolve here
    kernel = [1, 1, 1]
    video_timestamps = np.convolve(video_timestamps, kernel, mode="full")

    candidates = []
    # Connected Components
    active = False
    start = None

    
    for i, v in enumerate(video_timestamps):
        if active == False and v != 0:
            start = i
            active = True
        elif active == True and v == 0:
            candidates.append((start, i))
            active = False

    if active:
        candidates.append((start, len(video_timestamps)))

    return candidates
    

