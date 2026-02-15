def get_timestamps(times, video_length, gap_threshold=15):
    """
    Groups timestamps into variable-length segments.
    
    Args:
        times (list): List of timestamps (seconds) where matches occurred.
        video_length (int): Total duration of video.
        gap_threshold (int): Max seconds between frames to consider them the same segment.
                             Since we sample every 5s, a threshold of 15s allows for
                             continuous clips even if we miss 1-2 frames in between.
    
    Returns:
        list of tuples: [(start, end), (start, end), ...]
    """
    if not times:
        return []

    # specific sorting to ensure logic works
    times = sorted(list(set(times)))
    
    segments = []
    
    if not times:
        return []

    # Initialize first segment
    current_start = times[0]
    current_end = times[0]

    for t in times[1:]:
        # If this timestamp is close enough to the last one, extend the segment
        if t - current_end <= gap_threshold:
            current_end = t
        else:
            # Gap is too large, close current segment and start new one
            # Add a small buffer of padding (-2s start, +4s end) for context
            segments.append((max(0, current_start - 2), min(video_length, current_end + 4)))
            current_start = t
            current_end = t

    # Append the final segment
    segments.append((max(0, current_start - 2), min(video_length, current_end + 4)))

    return segments
    

