import cv2
import numpy as np
from scipy.linalg import hadamard
import os

print("--- Starting Robust Dual-Redundancy Authentication & Recovery ---")

# --- Configuration ---
block_size = 8
d = 8
RECOVERY_BLOCK_SHAPE = (2, 2)
AUTH_BITS = 16
SECRET_KEY_1 = 42
SECRET_KEY_2 = 999

# --- Helper Functions ---
def generate_chaotic_permutation(p1, p2, length=64):
    """Generates a pseudo-random permutation consistent with embedding."""
    sequence = np.zeros(length)
    g = p1
    for i in range(length):
        g = p2 * g * (1 - g)
        sequence[i] = g
    return np.argsort(sequence)

def bits_to_block(bits):
    """Reconstructs 8x8 block from 24 bits."""
    pixels = []
    # 24 bits -> 4 pixels (6 bits each)
    for i in range(0, 24, 6):
        # Join 6 bits to string, parse to int
        val = int("".join(map(str, bits[i:i+6])), 2)
        # Dequantize (0-63 -> 0-255 mapping approx)
        pixels.append((val << 2) + 2) 
    
    small = np.array(pixels, dtype=np.uint8).reshape(2,2)
    # Upscale back to 8x8 using Nearest Neighbor to keep sharp edges
    return cv2.resize(small, (8,8), interpolation=cv2.INTER_NEAREST)

def create_block_mapping(num_blocks, key):
    """Recreates the shuffling map used in embedding."""
    indices = np.arange(num_blocks)
    prng = np.random.RandomState(key)
    prng.shuffle(indices)
    return indices # This is P. Host_Index = P[Source_Index]

# --- Main Logic ---
try:
    # 1. Load Attacked Image
    img_path = 'lena_attacked.png'
    if not os.path.exists(img_path):
        print(f"Error: '{img_path}' not found. Please run attack_image.py first.")
        exit()
        
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    
    # Calculate block counts
    num_blocks_per_channel = (height // block_size) * (width // block_size)
    total_blocks = num_blocks_per_channel * 3
    channels = cv2.split(img)

    print(f"Analyzing Image: {width}x{height} ({total_blocks} blocks total)")

    # 2. Setup Matrices & Mappings
    H_d = hadamard(d)
    W_d = hadamard(d)
    M_d = np.bitwise_and(np.where(H_d > 0, 1, 0), np.where(W_d > 0, 1, 0))

    # P maps: Source_Index -> Host_Index
    P_map_1 = create_block_mapping(total_blocks, SECRET_KEY_1)
    P_map_2 = create_block_mapping(total_blocks, SECRET_KEY_2)

    # 3. Phase 1: Tamper Detection
    is_tampered = np.zeros(total_blocks, dtype=bool)
    extracted_payloads = {} 
    
    # Visualization map (White = Tampered)
    tamper_map_vis = np.zeros((height, width), dtype=np.uint8)
    
    # We need to track T2 groups to validate T2 checksums
    # Structure: t1_values[channel][block_idx]
    t1_values_extracted = [[0] * num_blocks_per_channel for _ in range(3)]

    print("Step 1: Analyzing Blocks & Extracting Payloads...")
    
    for total_idx in range(total_blocks):
        ch_id = total_idx // num_blocks_per_channel
        blk_idx = total_idx % num_blocks_per_channel
        
        row = blk_idx // (width // block_size)
        col = blk_idx % (width // block_size)
        y, x = row * block_size, col * block_size

        block = channels[ch_id][y:y+block_size, x:x+block_size]
        
        # A. Calculate Image Properties (from MSB)
        msb = np.bitwise_and(block, 254) # Clear LSB
        
        # Calculate T1 from current content
        S_d = (M_d @ msb @ M_d) / d
        T1_calc = (int(np.trace(S_d)) + 1) % (2**10)
        
        # Calculate Chaotic Keys
        mean_val = np.mean(msb)
        std_val = np.std(msb)
        p1 = (mean_val + 1) / 257.0
        p2 = min(4.0, 3.99 + (std_val - int(std_val)) * 0.1)
        
        # B. Extract LSB Payload using Chaotic Permutation
        perm = generate_chaotic_permutation(p1, p2, 64)
        flat = block.flatten() % 2 # Get LSBs
        
        payload = [0]*64
        for k in range(64):
            payload[k] = flat[perm[k]]
        
        extracted_payloads[total_idx] = payload
        
        # C. Verify T1 (Immediate Check)
        T1_bits = payload[:10]
        T1_extract = int("".join(map(str, T1_bits)), 2)
        
        t1_values_extracted[ch_id][blk_idx] = T1_extract # Save for T2 check later

        if T1_calc != T1_extract:
            is_tampered[total_idx] = True
            tamper_map_vis[y:y+block_size, x:x+block_size] = 255
            
    # D. Verify T2 (Group Checksum) - Secondary Validation
    # T2 validates groups of 5 blocks. If T2 fails, all 5 might be suspect.
    print("Step 2: Verifying T2 Group Checksums...")
    for ch_id in range(3):
        t2_counter = 0
        for blk_idx in range(num_blocks_per_channel):
            total_idx = ch_id * num_blocks_per_channel + blk_idx
            
            # Retrieve extracted T2 from payload
            payload = extracted_payloads[total_idx]
            T2_extract = int("".join(map(str, payload[10:16])), 2)
            
            # Determine which T2 group we belong to
            if blk_idx > 0 and blk_idx % 5 == 0:
                t2_counter += 1
            
            # Calculate what T2 *should* be based on extracted T1s
            group_start_idx = t2_counter * 5
            group_t1s = t1_values_extracted[ch_id][group_start_idx : group_start_idx + 5]
            
            # Only check if we have a full group
            if len(group_t1s) == 5:
                T2_calc = int((sum(group_t1s) / 5) % (2**6))
                
                # If T2 mismatch, mark this specific block as tampered
                # (Note: T1 is strictly local, T2 is regional)
                if T2_calc != T2_extract and not is_tampered[total_idx]:
                    is_tampered[total_idx] = True
                    # Update visualization
                    row = blk_idx // (width // block_size)
                    col = blk_idx % (width // block_size)
                    y, x = row * block_size, col * block_size
                    tamper_map_vis[y:y+block_size, x:x+block_size] = 255

    cv2.imwrite('tamper_map_detected.png', tamper_map_vis)
    print(f"Tamper Map Saved. Total Tampered Blocks: {np.sum(is_tampered)}")

    # 4. Phase 2: Recovery
    print("Step 3: Attempting Recovery (Dual Redundancy)...")
    recovered_channels = [c.copy() for c in channels]
    inpaint_mask = np.zeros((height, width), dtype=np.uint8)
    
    recovered_count = 0
    inpaint_count = 0

    for total_idx in range(total_blocks):
        if not is_tampered[total_idx]:
            continue # Block is authentic

        ch_id = total_idx // num_blocks_per_channel
        blk_idx = total_idx % num_blocks_per_channel
        row = blk_idx // (width // block_size)
        col = blk_idx % (width // block_size)
        y, x = row * block_size, col * block_size
        
        # We need to recover THIS block (Source).
        # We look for Hosts that store our data.
        
        # Try Host 1 (Primary Backup)
        host_idx_1 = P_map_1[total_idx]
        if not is_tampered[host_idx_1]:
            # Host 1 is valid! Extract Rec1 (Bits 16-40)
            payload = extracted_payloads[host_idx_1]
            rec_bits = payload[16 : 40]
            recovered_channels[ch_id][y:y+block_size, x:x+block_size] = bits_to_block(rec_bits)
            recovered_count += 1
            continue
            
        # Try Host 2 (Secondary Backup)
        host_idx_2 = P_map_2[total_idx]
        if not is_tampered[host_idx_2]:
            # Host 2 is valid! Extract Rec2 (Bits 40-64)
            payload = extracted_payloads[host_idx_2]
            rec_bits = payload[40 : 64]
            recovered_channels[ch_id][y:y+block_size, x:x+block_size] = bits_to_block(rec_bits)
            recovered_count += 1
            continue
            
        # Both failed -> Mark for Inpainting
        inpaint_mask[y:y+block_size, x:x+block_size] = 255
        # Set to black temporarily
        recovered_channels[ch_id][y:y+block_size, x:x+block_size] = 0
        inpaint_count += 1

    print(f"Recovery Stats: {recovered_count} blocks recovered cryptographically, {inpaint_count} blocks need inpainting.")

    # 5. Phase 3: Inpainting Fallback
    merged_recovered = cv2.merge(recovered_channels)
    
    if inpaint_count > 0:
        print("Step 4: Filling remaining gaps with Telea Inpainting...")
        # Radius 3px considers immediate 8-neighbors
        final_restored = cv2.inpaint(merged_recovered, inpaint_mask, 3, cv2.INPAINT_TELEA)
    else:
        final_restored = merged_recovered

    cv2.imwrite('recovered_final.png', final_restored)
    print("Process Complete. Result saved as 'recovered_final.png'.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()