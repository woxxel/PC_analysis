import os
import re

def extract_basename(folder_path):
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    pattern = re.compile(r'^(.*?)(\d{4})\.tif$')
    for filename in tif_files:
        match = pattern.match(filename)
        if match:
            return match.group(1)
    return None

def check_all_image_files_present(folder_path, max_frame):
    basename = extract_basename(folder_path)
    if not basename:
        print("No valid .tif files found to extract basename.")
        return None
    missing_frames = []
    for frame_num in range(max_frame + 1):
        filename = f"{basename}{frame_num:04d}.tif"
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            missing_frames.append(frame_num)
    if not missing_frames:
        return True
    else:
        return missing_frames

# Example usage:
if __name__ == "__main__":
    folder = input("Enter folder path: ")
    max_frame = int(input("Enter maximum frame number: "))
    result = check_all_image_files_present(folder, max_frame)
    if result is True:
        print("true")
    elif result is None:
        print("Could not extract basename.")
    else:
        print("Missing frame numbers:", result)