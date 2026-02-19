import os
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
CONFIG = {
    'RANDOM_STATE': 42,
    'MAX_ASSETS': 621,
    'NUM_SYNTHETIC_SAMPLES_BASE': 4000, 
    'ANOMALY_NOISE_PROB': 0.15,
    'MIXED_ANOMALY_PROB': 0.1,
}

ACOUSTIC_CONFIG = {
    'DATA_ROOT': r'c:\Users\Raza-SEAGUARD\Downloads\MAFT\DeepShip-main',
    'SHIP_TYPES': ['Cargo', 'Passengership', 'Tanker', 'Tug'],
    'SAMPLE_RATE': 44100,
    'N_FFT': 2048,
    'HOP_LENGTH': 512,
    'N_MELS': 128,
    'DURATION': 5.0
}

ENVIRONMENTAL_CONFIG = {
    'SEA_STATE_RANGE': (0, 6),
    'CLOUD_COVER_RANGE': (0.0, 0.8),
    'TIME_OF_DAY_OPTIONS': ['day', 'dusk', 'night'],
    'WAKE_PROBABILITY': 0.8,
}

AIS_CONFIG = {
    'INPUT_DIM': 12,
    'USE_TRAJECTORY_HISTORY': True,
    'HISTORY_LENGTH': 5
}

VESSEL_TYPE_ENCODING = {
    'Cargo': 0.25,
    'Passengership': 0.5,
    'Tanker': 0.75,
    'Tug': 1.0,
    'unknown': 0.5
}

# Local Paths
BASE_DIR = r'c:\Users\Raza-SEAGUARD\Downloads\MAFT'
SYNTHETIC_DATA_DIR = os.path.join(BASE_DIR, "synthetic_data_v2")
DATA_ROOT_IMG = os.path.join(BASE_DIR, 'Ship Detection from Aerial Images')
IMAGE_DIR = os.path.join(DATA_ROOT_IMG, 'images')
ANNOTATION_DIR = os.path.join(DATA_ROOT_IMG, 'annotations')
MASKS_DIR = os.path.join(BASE_DIR, 'working', 'masks')
SHIP_ASSETS_NORM_DIR = os.path.join(BASE_DIR, 'working', 'ship_assets_normalised')
AIS_FILE_PATH = os.path.join(BASE_DIR, 'AIS Data For Ships', 'AIS_2022_03_31.csv')
METADATA_PATH = os.path.join(SYNTHETIC_DATA_DIR, "metadata.csv")

os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(SHIP_ASSETS_NORM_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)

random.seed(CONFIG['RANDOM_STATE'])
np.random.seed(CONFIG['RANDOM_STATE'])

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def normalise_asset_orientation(asset_pil):
    arr = np.array(asset_pil)
    alpha = arr[:, :, 3]
    y, x = np.where(alpha > 0)
    if len(x) == 0:
        return asset_pil
    points = np.column_stack((x, y)).astype(np.float32)
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    angle = np.arctan2(main_axis[1], main_axis[0])
    rot_deg = -np.degrees(angle)
    rotated = asset_pil.rotate(rot_deg, expand=True, resample=Image.BICUBIC)
    rot_arr = np.array(rotated)
    rot_alpha = rot_arr[:, :, 3]
    y_rot, x_rot = np.where(rot_alpha > 0)
    if len(x_rot) == 0:
        return rotated
    com_x = np.mean(x_rot)
    img_center_x = rotated.width / 2
    if com_x < img_center_x:
        rotated = rotated.transpose(Image.FLIP_LEFT_RIGHT)
    return rotated

def simulate_sar_realistic(aerial_img, speckle_type='gamma', looks=4):
    """Simulates SAR with multiplicative speckle and backscatter physics."""
    gray = aerial_img.convert('L')
    arr = np.array(gray, dtype=np.float32) / 255.0
    
    # Sigmoid-based backscatter mapping for better intensity transition
    backscatter = 0.15 + 0.65 / (1 + np.exp(-10 * (arr - 0.4)))
    
    if speckle_type == 'gamma':
        # Multiplicative gamma noise (fully developed speckle model)
        speckle = np.random.gamma(looks, 1/looks, arr.shape)
    else:
        speckle = np.random.exponential(1.0, arr.shape)
        
    sar_intensity = backscatter * speckle
    
    # Log transformation (standard SAR dynamic range compression)
    sar_db = 10 * np.log10(sar_intensity + 1e-6)
    sar_norm = (sar_db - sar_db.min()) / (sar_db.max() - sar_db.min() + 1e-10)
    
    sar_img = (sar_norm * 255).astype(np.uint8)
    return Image.fromarray(sar_img).convert('RGB')

def add_sar_artifacts(sar_img):
    arr = np.array(sar_img.convert('L'), dtype=np.float32)
    streak_intensity = np.random.uniform(5, 15)
    for i in range(arr.shape[0]):
        if np.random.random() < 0.1:
            arr[i, :] = np.clip(arr[i, :] + streak_intensity, 0, 255)
    return Image.fromarray(arr.astype(np.uint8)).convert('RGB')

def generate_dynamic_background(size=(768, 768), sea_state=2, cloud_cover=0.0, time_of_day='day'):
    base_colors = {'day': (25, 40, 55), 'dusk': (40, 35, 50), 'night': (10, 15, 25)}
    base_color = base_colors.get(time_of_day, (25, 40, 55))
    bg = Image.new('RGB', size, base_color)
    arr = np.array(bg, dtype=np.float32)

    if sea_state > 0:
        wave_amplitude = sea_state * 4
        wave_frequency = 0.02 + sea_state * 0.005
        x = np.arange(size[0]); y = np.arange(size[1])
        X, Y = np.meshgrid(x, y)
        
        # Multi-scale fractal-like wave pattern
        wave_pattern = (np.sin(X * wave_frequency) * wave_amplitude +
                        # High-freq ripples
                        np.sin(Y * wave_frequency * 2.5) * wave_amplitude * 0.4 +
                        # Cross-sea interference
                        np.sin((X + Y) * 0.1) * 2.0 +
                        # Perlin-like random texture (noise)
                        np.random.normal(0, 1.5, size))
        
        wave_shading = (wave_pattern + wave_amplitude * 3) / (wave_amplitude * 6)
        for c in range(3):
            arr[:, :, c] = np.clip(arr[:, :, c] + wave_shading * 20 - 10, 0, 255)

    if sea_state <= 5 and random.random() < ENVIRONMENTAL_CONFIG['WAKE_PROBABILITY']:
        arr = add_ship_wake(arr, wake_intensity=max(0, 5-sea_state))

    if cloud_cover > 0:
        arr = add_cloud_cover(arr, cloud_cover)

    return Image.fromarray(arr.astype(np.uint8))

def add_ship_wake(arr, wake_intensity=3):
    h, w = arr.shape[:2]
    wake_center_x = w // 2
    wake_start_y = h // 2 + 20
    wake_color = 30 * wake_intensity / 5
    for offset in range(50, min(h - wake_start_y, 200)):
        spread = int(offset * 0.3)
        alpha = max(0, 1 - offset / 200)
        left_x = max(0, wake_center_x - spread)
        right_x = min(w-1, wake_center_x + spread)
        y = wake_start_y + offset
        if y < h:
            arr[y, left_x] = np.clip(arr[y, left_x] + wake_color * alpha, 0, 255)
            arr[y, right_x] = np.clip(arr[y, right_x] + wake_color * alpha, 0, 255)
    return arr

def add_cloud_cover(arr, cloud_cover):
    h, w = arr.shape[:2]
    num_clouds = int(cloud_cover * 10)
    for _ in range(num_clouds):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        cloud_size = random.randint(50, 200)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        cloud_mask = dist < cloud_size
        arr[cloud_mask] = arr[cloud_mask] * 0.7
    return arr

def encode_ais_enhanced(ais_point, trajectory_history=None, vessel_metadata=None):
    features = []
    sog = min(ais_point['SOG'] / 30.0, 1.0)
    cog = ais_point['COG'] / 360.0
    signal_present = float(ais_point.get('signal_present', 1.0))
    features.extend([sog, cog, signal_present])

    if trajectory_history and len(trajectory_history) >= 2:
        cog_history = [p['COG'] for p in trajectory_history[-AIS_CONFIG['HISTORY_LENGTH']:]]
        sog_history = [p['SOG'] for p in trajectory_history[-AIS_CONFIG['HISTORY_LENGTH']:]]
        cog_changes = [abs(((cog_history[i] - cog_history[i-1]) + 180) % 360 - 180)
                       for i in range(1, len(cog_history))]
        rate_of_turn = np.mean(cog_changes) / 10.0 if cog_changes else 0.0
        trajectory_consistency = 1.0 - (np.std(cog_history) / 180.0)
        speed_variance = min(np.std(sog_history) / 10.0, 1.0)
    else:
        rate_of_turn = 0.0
        trajectory_consistency = 1.0
        speed_variance = 0.0
    features.extend([rate_of_turn, trajectory_consistency, speed_variance])

    time_since_update = ais_point.get('time_since_update', 0) / 600.0
    features.append(min(time_since_update, 1.0))

    if vessel_metadata:
        vessel_type_encoded = VESSEL_TYPE_ENCODING.get(vessel_metadata.get('type', 'unknown'), 0.5)
        vessel_length_proxy = min(vessel_metadata.get('length', 100) / 400.0, 1.0)
    else:
        vessel_type_encoded = 0.5
        vessel_length_proxy = 0.25
    features.extend([vessel_type_encoded, vessel_length_proxy])

    nav_status = ais_point.get('nav_status', 0) / 15.0
    proximity_to_shore = min(ais_point.get('proximity_to_shore', 50) / 100.0, 1.0)
    features.extend([nav_status, proximity_to_shore])
    # Inject AIS noise (Reviewer 2 Concern 5)
    if random.random() < 0.2:
        features[0] *= (1.0 + np.random.normal(0, 0.05)) # 5% SOG noise
        features[1] = (features[1] + np.random.normal(0, 0.01)) % 1.0 # 1% COG drift
        features[3:6] += np.random.normal(0, 0.02, 3) # noise in kinematic derivatives

    return np.array(features, dtype=np.float32)

def add_faint_vessel(background, opacity=0.2):
    arr = np.array(background)
    h, w = arr.shape[:2]
    vessel_size = random.randint(15, 30)
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    vessel_mask = dist < vessel_size
    arr[vessel_mask] = arr[vessel_mask] * (1 - opacity) + 100 * opacity
    return Image.fromarray(arr.astype(np.uint8))

def add_multiple_vessels(background, count=2):
    arr = np.array(background)
    h, w = arr.shape[:2]
    for _ in range(count):
        vessel_size = random.randint(20, 40)
        cx = random.randint(w//4, 3*w//4)
        cy = random.randint(h//4, 3*h//4)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        vessel_mask = dist < vessel_size
        arr[vessel_mask] = np.clip(arr[vessel_mask] + 50, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def process_audio(path):
    try:
        y, sr = librosa.load(path, sr=ACOUSTIC_CONFIG['SAMPLE_RATE'],
                              duration=ACOUSTIC_CONFIG['DURATION'])
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_fft=ACOUSTIC_CONFIG['N_FFT'],
            hop_length=ACOUSTIC_CONFIG['HOP_LENGTH'],
            n_mels=ACOUSTIC_CONFIG['N_MELS']
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        img = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        img_3_channel = np.stack([img, img, img], axis=-1)
        return Image.fromarray((img_3_channel * 255).astype(np.uint8))
    except (Exception, librosa.util.exceptions.LibrosaError):
        return Image.new('RGB', (128, 128), 'black')

# ------------------------------------------------------------------------------
# MAIN GENERATION LOOP
# ------------------------------------------------------------------------------
def generate_enhanced_dataset():
    print("\n" + "="*70)
    print(f"Enhanced Dataset Generation – {CONFIG['NUM_SYNTHETIC_SAMPLES_BASE']} base samples → {CONFIG['NUM_SYNTHETIC_SAMPLES_BASE']*4} total")
    print("="*70)

    # Generate masks from XML (if needed)
    if not os.listdir(MASKS_DIR):
        print("Generating masks from annotations...")
        ann_files = [f for f in os.listdir(ANNOTATION_DIR) if f.endswith('.xml')]
        for ann in tqdm(ann_files, desc="Masks"):
            try:
                tree = ET.parse(os.path.join(ANNOTATION_DIR, ann))
                root = tree.getroot()
                img_file = root.find('filename').text
                img_path = os.path.join(IMAGE_DIR, img_file)
                if not os.path.exists(img_path): continue
                img = Image.open(img_path)
                mask = Image.new('L', img.size, 0)
                draw = ImageDraw.Draw(mask)
                for obj in root.findall('object'):
                    bnd = obj.find('bndbox')
                    xmin = int(float(bnd.find('xmin').text))
                    ymin = int(float(bnd.find('ymin').text))
                    xmax = int(float(bnd.find('xmax').text))
                    ymax = int(float(bnd.find('ymax').text))
                    draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
                mask.save(os.path.join(MASKS_DIR, os.path.splitext(img_file)[0]+'.png'))
            except Exception:
                continue

    # Extract and normalise ship assets
    print("\nExtracting and normalising ship assets...")
    img_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png','.jpg'))]
    count = 0
    if not os.listdir(SHIP_ASSETS_NORM_DIR):
        for img_file in tqdm(img_files[:CONFIG['MAX_ASSETS']], desc="Assets"):
            if count >= CONFIG['MAX_ASSETS']: break
            mask_path = os.path.join(MASKS_DIR, os.path.splitext(img_file)[0]+'.png')
            if not os.path.exists(mask_path): continue
            try:
                img = Image.open(os.path.join(IMAGE_DIR, img_file)).convert('RGBA')
                mask = Image.open(mask_path).convert('L')
                if np.array(mask).sum() == 0: continue
                img.putalpha(mask)
                bbox = img.getbbox()
                if bbox is None: continue
                ship = img.crop(bbox)
                if ship.size[0] <= 0 or ship.size[1] <= 0: continue
                norm_ship = normalise_asset_orientation(ship)
                norm_ship.save(os.path.join(SHIP_ASSETS_NORM_DIR, f'asset_{count}.png'))
                count += 1
            except Exception:
                continue

    asset_paths = [os.path.join(SHIP_ASSETS_NORM_DIR, f) for f in os.listdir(SHIP_ASSETS_NORM_DIR)]
    print(f"Using {len(asset_paths)} normalised assets.")

    # Collect acoustic files
    acoustic_files = {t: [] for t in ACOUSTIC_CONFIG['SHIP_TYPES']}
    for t in ACOUSTIC_CONFIG['SHIP_TYPES']:
        d = os.path.join(ACOUSTIC_CONFIG['DATA_ROOT'], t)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.wav','.mp3')):
                    acoustic_files[t].append(os.path.join(d, f))
        print(f"Found {len(acoustic_files[t])} audio files for '{t}'")

    # Map assets to ship types randomly
    asset_type = {p: random.choice(ACOUSTIC_CONFIG['SHIP_TYPES']) for p in asset_paths}

    # Create output directories
    for mod in ['aerial','sar','acoustic']:
        os.makedirs(os.path.join(SYNTHETIC_DATA_DIR, mod), exist_ok=True)

    # Load AIS
    try:
        ais_df = pd.read_csv(AIS_FILE_PATH, usecols=['MMSI','BaseDateTime','LAT','LON','SOG','COG'])
        ais_df.fillna({'SOG':0,'COG':0}, inplace=True)
        valid_ais = ais_df[ais_df['SOG']>1].nlargest(6000, 'SOG')
    except Exception as e:
        print(f"Warning: Could not load AIS data from {AIS_FILE_PATH} ({e}). Using fallback synthetic AIS.")
        valid_ais = pd.DataFrame({'SOG': np.random.uniform(5,25,6000), 'COG': np.random.uniform(0,360,6000)})

    ship_assets = [Image.open(p).convert('RGBA') for p in asset_paths if os.path.exists(p)]

    metadata = []
    for i in tqdm(range(CONFIG['NUM_SYNTHETIC_SAMPLES_BASE']), desc="Generating"):
        asset_idx = i % len(asset_paths)
        asset = ship_assets[asset_idx]
        ship_type = asset_type[asset_paths[asset_idx]]

        if not acoustic_files[ship_type]:
            continue
        audio_path = random.choice(acoustic_files[ship_type])
        ac_img = process_audio(audio_path)
        ac_img.save(os.path.join(SYNTHETIC_DATA_DIR, 'acoustic', f"{i}.png"))

        # Random AIS point
        ais_point = valid_ais.sample(1, random_state=CONFIG['RANDOM_STATE']+i).iloc[0].to_dict()

        # Environmental conditions
        sea_state = random.randint(*ENVIRONMENTAL_CONFIG['SEA_STATE_RANGE'])
        cloud_cover = random.uniform(*ENVIRONMENTAL_CONFIG['CLOUD_COVER_RANGE'])
        time_of_day = random.choice(ENVIRONMENTAL_CONFIG['TIME_OF_DAY_OPTIONS'])
        background = generate_dynamic_background(sea_state=sea_state, cloud_cover=cloud_cover, time_of_day=time_of_day)

        base_ais = {
            'SOG': ais_point['SOG'],
            'COG': ais_point['COG'],
            'signal_present': 1.0,
            'nav_status': random.randint(0,15),
            'proximity_to_shore': random.uniform(0,100)
        }

        for j, anom in enumerate(['Correlated','Dark Vessel','Spoofing','Kinematic Anomaly']):
            sample_id = i*4 + j
            label = f"{anom}-{ship_type}"

            if anom == 'Correlated':
                angle = -(ais_point['COG'] - 90)
                rot = asset.rotate(angle, expand=True, resample=Image.BICUBIC)
                bg_w, bg_h = background.size
                pos = ((bg_w - rot.width)//2, (bg_h - rot.height)//2)
                aerial = background.copy()
                aerial.paste(rot, pos, rot)
                ais_data = base_ais

            elif anom == 'Dark Vessel':
                angle = -(ais_point['COG'] - 90)
                rot = asset.rotate(angle, expand=True, resample=Image.BICUBIC)
                bg_w, bg_h = background.size
                pos = ((bg_w - rot.width)//2, (bg_h - rot.height)//2)
                aerial = background.copy()
                aerial.paste(rot, pos, rot)
                if random.random() < 0.2:
                    ais_data = {'SOG': random.uniform(0.1,2.0), 'COG': random.uniform(0,360), 'signal_present':0.3}
                else:
                    ais_data = {'SOG':0, 'COG':0, 'signal_present':0}

            elif anom == 'Spoofing':
                r = random.random()
                if r < 0.15:
                    aerial = add_faint_vessel(background, opacity=0.2)
                elif r < 0.3:
                    aerial = generate_dynamic_background(cloud_cover=0.7)
                else:
                    aerial = background
                ais_data = base_ais

            else:  # Kinematic Anomaly
                r = random.random()
                if r < 0.4:
                    disp = random.randint(30, 89)
                else:
                    disp = random.randint(90, 180)
                fake_cog = (ais_point['COG'] + disp) % 360
                angle = -(fake_cog - 90)
                rot = asset.rotate(angle, expand=True, resample=Image.BICUBIC)
                bg_w, bg_h = background.size
                pos = ((bg_w - rot.width)//2, (bg_h - rot.height)//2)
                aerial = background.copy()
                aerial.paste(rot, pos, rot)
                ais_data = base_ais

            # Generate SAR
            sar = simulate_sar_realistic(aerial, speckle_type='gamma', looks=4)
            sar = add_sar_artifacts(sar)

            # Save images
            aerial.save(os.path.join(SYNTHETIC_DATA_DIR, 'aerial', f"{sample_id}.png"))
            sar.save(os.path.join(SYNTHETIC_DATA_DIR, 'sar', f"{sample_id}.png"))

            # Encode AIS (12‑dim)
            traj = [ais_data]*3
            vessel_meta = {'type': ship_type, 'length': random.randint(50,300)}
            ais_feat = encode_ais_enhanced(ais_data, traj, vessel_meta)

            metadata.append({
                'sample_id': sample_id,
                'group_id': i,
                'label': label,
                'anomaly_type': anom,
                'ship_type': ship_type,
                'aerial_path': os.path.join('aerial', f"{sample_id}.png"),
                'sar_path': os.path.join('sar', f"{sample_id}.png"),
                'acoustic_path': os.path.join('acoustic', f"{i}.png"),
                'sog': float(ais_feat[0]),
                'cog': float(ais_feat[1]),
                'signal_present': float(ais_feat[2]),
                'rate_of_turn': float(ais_feat[3]),
                'trajectory_consistency': float(ais_feat[4]),
                'speed_variance': float(ais_feat[5]),
                'time_since_update': float(ais_feat[6]),
                'vessel_type_encoded': float(ais_feat[7]),
                'vessel_length_proxy': float(ais_feat[8]),
                'nav_status': float(ais_feat[9]),
                'proximity_to_shore': float(ais_feat[10]),
            })

    df = pd.DataFrame(metadata)
    df.to_csv(METADATA_PATH, index=False)
    print(f"\nSaved {len(df)} samples to {SYNTHETIC_DATA_DIR}")
    return df

if __name__ == "__main__":
    generate_enhanced_dataset()
    print("\n✅ Enhanced dataset generation complete.")
