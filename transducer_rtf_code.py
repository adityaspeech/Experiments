import time
import os
import csv
from pydub import AudioSegment
from birch_stt_server.offline_asr import init_recognizer

# Initialize the recognizer
cfg, recognizer = init_recognizer('/data/models/onnx-fp32-rnnt5-61K-0907-checkpoint-657000-avg-7-stt2.1.0')

# Input directory containing folders with WAV files
input_path = "/home/avinash.r/rtf_calculation_4_segmented_audios/"

# Output directory where CSV files will be saved (modify as needed)
output_path = "/home/avinash.r/rtf_results_segmented_4_audios_transducer"  # Change this to your desired output directory

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Process each folder in the input directory
for folder_name in os.listdir(input_path):
    folder_path = os.path.join(input_path, folder_name)
    
    # Skip if not a directory
    if not os.path.isdir(folder_path):
        continue
    
    # Prepare data list for this folder
    folder_data = []
    
    # Process each WAV file in the folder
    for wav_file in os.listdir(folder_path):
        if not wav_file.lower().endswith('.wav'):
            continue
            
        wav_path = os.path.join(folder_path, wav_file)
        
        try:
            # Start timing for decoding
            start_time = time.time()
            
            # Load audio file
            audio = AudioSegment.from_file(wav_path)
            duration = len(audio) / 1000  # Duration in seconds
            
            # Perform transcription
            transcription_result = recognizer.decode_audio(audio)
            
            # Calculate decoding time and RTF
            decoding_time = time.time() - start_time
            rtf = decoding_time / duration if duration > 0 else 0
            
            if transcription_result:
                for segment in transcription_result:
                    text = segment.get('text', "").strip()
                    
                    folder_data.append({
                        'Folder Name': folder_name,
                        'WAV File Name': wav_file,
                        'Text': text,
                        'Duration (s)': round(duration, 2),
                        'Decoding Time (s)': round(decoding_time, 2),
                        'RTF': round(rtf, 4)
                    })
            
            print(f"Processed {wav_file} in folder {folder_name}")
            
        except Exception as e:
            print(f"Failed to process {wav_path}: {e}")
    
    # Create CSV file for this folder if we have data
    if folder_data:
        csv_filename = os.path.join(output_path, f"{folder_name}.csv")  # Save in output_path
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Folder Name', 'WAV File Name', 'Text', 'Duration (s)', 'Decoding Time (s)', 'RTF']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in folder_data:
                writer.writerow(row)
        
        print(f"Created CSV file: {csv_filename}")
    else:
        print(f"No valid WAV files processed in folder {folder_name}")
