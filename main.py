import librosa
import soundfile as sf
import json
import os
from audiomentations import Compose, AddBackgroundNoise, PitchShift, TimeStretch, Gain


# Augmentation Settings. You can add any settings that you wish
augment = Compose([
    AddBackgroundNoise(sounds_path="background_noise", min_snr_db=10, max_snr_db=20, p=1),
    PitchShift(min_semitones=0.5, max_semitones=1, p=1),
    TimeStretch(min_rate=1, max_rate=1.25, p=1) ,
    Gain(min_gain_db=-1, max_gain_db=1, p=1.0)

])


# Method which will write metadata of our audio
def save_metadata(file_name, metadata):
    metadata_file = file_name.replace(".mp3", ".json")
    with open(f"augmented/{metadata_file}", "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":

    # Path to sounds
    path = "audio"
    for file_name in os.listdir(path):
        # Checking if our path contains mp3 files, you can add more formats of sound
        if file_name.endswith("mp3"):
            # Loading sound data
            signal, sr = librosa.load(f"audio/{file_name}")

            metadata = {
                "file_name": file_name,
                "sample_rate": sr,
                "augmentations": []
            }
            # Applying augmentation
            augmented_signal = augment(signal, sr)
            # Going through transforms that we applied to sound
            for transform in augment.transforms:
                # Create information about transformation
                augmentation_info = {
                    "name": transform.__class__.__name__,
                    "parameters": {}
                }
                # Checking Transformation name and adding data about transformation
                if isinstance(transform, AddBackgroundNoise):
                    augmentation_info["parameters"] = {
                        "min_amplitude": transform.min_snr_db,
                        "max_amplitude": transform.max_snr_db
                    }
                elif isinstance(transform, PitchShift):
                    augmentation_info["parameters"] = {
                        "min_semitones": transform.min_semitones,
                        "max_semitones": transform.max_semitones
                    }
                elif isinstance(transform, TimeStretch):
                    augmentation_info["parameters"] = {
                        "min_rate": transform.min_rate,
                        "max_rate": transform.max_rate
                    }
                elif isinstance(transform, Gain):
                    augmentation_info["parameters"] = {
                        "min_gain_in_db": transform.min_gain_db,
                        "max_gain_in_db": transform.max_gain_db
                    }
                # Adding to our metadata variable our transformations
                metadata["augmentations"].append(augmentation_info)
            # Creating dir if it does not exist
            os.makedirs("augmented", exist_ok=True)
            # writing mew sounds
            sf.write(f"augmented/{file_name}", augmented_signal, sr)
            # saving metadata
            save_metadata(file_name, metadata)