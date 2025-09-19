import cv2
import argparse
import numpy as np
from openvino.runtime import Core
import csv
import json
from collections import Counter
from pathlib import Path


# Load Kinetics-600 labels
with open("kinetics_600_labels.json", "r") as f:
    KINETICS_600_LABELS = json.load(f)

def get_action_name(action_id):
    return KINETICS_600_LABELS.get(str(action_id), f"action_{action_id}")

# Model paths
BASE_DIR = Path(__file__).parent.resolve()
ENCODER_XML = BASE_DIR / "models/intel_action/encoder/FP32/action-recognition-0001-encoder.xml"
ENCODER_BIN = BASE_DIR / "models/intel_action/encoder/FP32/action-recognition-0001-encoder.bin"
DECODER_XML = BASE_DIR / "models/intel_action/decoder/FP32/action-recognition-0001-decoder.xml"
DECODER_BIN = BASE_DIR / "models/intel_action/decoder/FP32/action-recognition-0001-decoder.bin"

SEQUENCE_LENGTH = 16

def load_models(device="AUTO"):
    ie = Core()
    dev = "GPU" if "GPU" in ie.available_devices else "CPU"
    if device != "AUTO":
        dev = device
    print(f"Using device: {dev}")

    encoder_model = ie.read_model(model=ENCODER_XML, weights=ENCODER_BIN)
    decoder_model = ie.read_model(model=DECODER_XML, weights=DECODER_BIN)

    compiled_encoder = ie.compile_model(model=encoder_model, device_name=dev)
    compiled_decoder = ie.compile_model(model=decoder_model, device_name=dev)

    return compiled_encoder, compiled_encoder.input(0), compiled_encoder.output(0), \
           compiled_decoder, compiled_decoder.input(0), compiled_decoder.output(0)

def preprocess_frame(frame, input_shape):
    N, C, H, W = input_shape
    h, w = frame.shape[:2]

    scale = min(W/w, H/h)
    new_w, new_h = int(w*scale), int(h*scale)
    frame_resized = cv2.resize(frame, (new_w, new_h))

    pad_top = (H - new_h) // 2
    pad_bottom = H - new_h - pad_top
    pad_left = (W - new_w) // 2
    pad_right = W - new_w - pad_left
    frame_padded = cv2.copyMakeBorder(frame_resized, pad_top, pad_bottom, pad_left, pad_right,
                                      borderType=cv2.BORDER_CONSTANT, value=[0,0,0])

    frame_padded = frame_padded.transpose(2, 0, 1)
    return np.expand_dims(frame_padded, axis=0).astype(np.float32)

def detect_action_sequences(all_actions, min_duration=2.0):
    if not all_actions:
        return []
    sequences = []
    current_sequence = None
    for timestamp, frame_id, action_id, score, action_name in sorted(all_actions, key=lambda x: x[0]):
        if current_sequence and (action_id == current_sequence['action_id'] and timestamp - current_sequence['end_time'] < 3.0):
            current_sequence['end_time'] = timestamp
            current_sequence['max_score'] = max(current_sequence['max_score'], score)
            current_sequence['frames'].append(frame_id)
        else:
            if current_sequence and (current_sequence['end_time'] - current_sequence['start_time'] >= min_duration):
                sequences.append(current_sequence)
            current_sequence = {'action_id': action_id, 'action_name': action_name,
                                'start_time': timestamp, 'end_time': timestamp, 'max_score': score,
                                'frames': [frame_id]}
    if current_sequence and (current_sequence['end_time'] - current_sequence['start_time'] >= min_duration):
        sequences.append(current_sequence)
    return sequences

def run_action_detection(video_path, device="AUTO", sample_rate=30, log_file="action_log.csv", 
                        debug=False, top_k=8, confidence_threshold=0.01, show_video=False,
                        interesting_actions=None, progress_callback=None, cancel_flag=None):
    """
    interesting_actions: list of action names to detect (optional).
    progress_callback: function(progress, total, stage, message)
    cancel_flag: threading.Event() or similar to cancel processing
    Returns actions sorted by confidence (highest first)
    """
    if interesting_actions is not None:
        interesting_actions_set = set([s.lower() for s in interesting_actions])
    else:
        interesting_actions_set = None

    compiled_encoder, encoder_input, encoder_output, compiled_decoder, decoder_input, decoder_output = load_models(device)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sequence_buffer = []
    all_actions = []

    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_mmss", "frame_id", "action_id", "action_name", "score", "timestamp_seconds"])

        frame_id = 0
        while True:
            # Check for cancellation
            if cancel_flag and cancel_flag.is_set():
                print("⚠️ Action detection canceled by user.")
                break

            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if frame_id % sample_rate != 0:
                continue

            timestamp_secs = frame_id / fps
            mins, secs = divmod(int(timestamp_secs), 60)
            timestamp_str = f"{mins:02d}:{secs:02d}"

            clip = preprocess_frame(frame, encoder_input.shape)
            features = compiled_encoder([clip])[encoder_output].squeeze()
            sequence_buffer.append(features)
            if len(sequence_buffer) > SEQUENCE_LENGTH:
                sequence_buffer.pop(0)

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                sequence_array = np.expand_dims(np.stack(sequence_buffer, axis=0), axis=0)
                predictions = compiled_decoder([sequence_array])[decoder_output].flatten()

                top_indices = np.argsort(predictions)[-top_k:][::-1]
                for idx in top_indices:
                    score = float(predictions[idx])
                    action_name = get_action_name(idx)
                    if score >= confidence_threshold:
                        if (interesting_actions_set is None) or (action_name.lower() in interesting_actions_set):
                            writer.writerow([timestamp_str, frame_id, idx, action_name, score, timestamp_secs])
                            all_actions.append((timestamp_secs, frame_id, idx, score, action_name))
                            if debug:
                                print(f"DEBUG: {timestamp_str} -> {action_name} (ID:{idx}, score:{score:.3f})")

            if show_video:
                cv2.imshow('Action Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Update GUI progress
            if progress_callback is not None:
                progress_callback(frame_id, total_frames, "Pipeline", "Running action recognition...")

    cap.release()
    if show_video:
        cv2.destroyAllWindows()
    print(f"All actions logged to {log_file}")

    if not all_actions:
        return []
  
    # Sort by confidence score (highest first) and return
    all_actions_by_confidence = sorted(all_actions, key=lambda x: x[3], reverse=True)
    print(f"Returning {len(all_actions_by_confidence)} actions sorted by confidence")
    print(f"Confidence range: {all_actions_by_confidence[0][3]:.3f} to {all_actions_by_confidence[-1][3]:.3f}")
    
    return all_actions_by_confidence

def print_top_actions(all_actions, top_n=20):
    sorted_actions = sorted(all_actions, key=lambda x: x[3], reverse=True)
    print(f"\nTop {min(top_n, len(sorted_actions))} actions (by confidence):")
    for i, (timestamp, frame_id, action_id, score, action_name) in enumerate(sorted_actions[:top_n]):
        mins, secs = divmod(int(timestamp), 60)
        print(f"{i+1:2d}. {mins:02d}:{secs:02d} -> {action_name} (score:{score:.3f})")

def print_most_common_actions(all_actions, top_n=20):
    counter = Counter([a[4] for a in all_actions])
    print(f"\nTop {min(top_n, len(counter))} most common actions:")
    for i, (action_name, count) in enumerate(counter.most_common(top_n)):
        print(f"{i+1:2d}. {action_name} ({count} occurrences)")

def print_action_sequences(all_actions):
    sequences = detect_action_sequences(all_actions)
    print(f"\nAction sequences detected ({len(sequences)}):")
    for i, seq in enumerate(sequences):
        duration = seq['end_time'] - seq['start_time']
        start_mins, start_secs = divmod(int(seq['start_time']), 60)
        end_mins, end_secs = divmod(int(seq['end_time']), 60)
        print(f"{i+1:2d}. {seq['action_name']} Duration: {duration:.1f}s ({start_mins:02d}:{start_secs:02d} - {end_mins:02d}:{end_secs:02d}) Max score: {seq['max_score']:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--device", type=str, default="AUTO")
    parser.add_argument("--sample-rate", type=int, default=30)
    parser.add_argument("--log-file", type=str, default="action_log.csv")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.01)
    args = parser.parse_args()

    results = run_action_detection(
        video_path=args.input,
        device=args.device,
        sample_rate=args.sample_rate,
        log_file=args.log_file,
        debug=args.debug,
        top_k=args.top_k,
        confidence_threshold=args.confidence,
        show_video=args.show_video
    )

    print_top_actions(results)
    print_most_common_actions(results)
    print_action_sequences(results)
