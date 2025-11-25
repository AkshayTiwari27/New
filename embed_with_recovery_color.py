import cv2
import numpy as np
from scipy.linalg import hadamard
import sys

print("--- Starting Dual-Redundancy Color Watermark Embedding ---")

# --- Configuration ---
block_size = 8
d = 8

# --- Recovery Configuration ---
RECOVERY_BLOCK_SHAPE = (2, 2)
RECOVERY_BITS_PER_PIXEL = 6
# 2x2 pixels * 6 bits = 24 bits of recovery data
RECOVERY_PAYLOAD_SIZE = RECOVERY_BLOCK_SHAPE[0] * RECOVERY_BLOCK_SHAPE[1] * RECOVERY_BITS_PER_PIXEL

# Authentication Bits: T1 (10) + T2 (6) = 16 bits
AUTH_BITS = 16 

# Total Capacity Needed: Auth (16) + Backup 1 (24) + Backup 2 (24) = 64 bits
# An 8x8 block has 64 pixels, so we use 1 bit per pixel (LSB). Perfect fit.
TOTAL_BITS = AUTH_BITS + RECOVERY_PAYLOAD_SIZE + RECOVERY_PAYLOAD_SIZE

SECRET_KEY_1 = 42   # Key for First Backup Map
SECRET_KEY_2 = 999  # Key for Second Backup Map

# --- Helper Functions ---

def generate_chaotic_permutation(p1, p2, length=64):
    """Generates a pseudo-random permutation of indices 0 to length-1."""
    sequence = np.zeros(length)
    g = p1
    for i in range(length):
        g = p2 * g * (1 - g)
        sequence[i] = g
    return np.argsort(sequence)

def create_recovery_bits(block):
    """Creates a 2x2 down-sampled, 6-bit quantized version of the 8x8 block."""
    # Resize to 2x2
    recovery_block = cv2.resize(block, RECOVERY_BLOCK_SHAPE, interpolation=cv2.INTER_AREA)
    # Quantize to 6 bits (0-63) by shifting right 2 bits
    quantized_block = recovery_block >> 2 
    bit_string = ""
    for pixel in quantized_block.flatten():
        bit_string += f"{pixel:06b}"
    return [int(bit) for bit in bit_string]

def create_block_mapping(num_blocks, key):
    """Creates a shuffle mapping for block indices."""
    indices = np.arange(num_blocks)
    prng = np.random.RandomState(key)
    prng.shuffle(indices)
    P = indices
    P_inv = np.empty_like(P)
    P_inv[P] = np.arange(num_blocks)
    return P, P_inv

# --- Main Logic ---
try:
    # 1. Load Image
    host_image = cv2.imread('4.1.04.tiff', cv2.IMREAD_COLOR)
    if host_image is None:
        # Fallback for testing if tiff not found, try png
        host_image = cv2.imread('lena.png', cv2.IMREAD_COLOR)
        if host_image is None:
            raise FileNotFoundError("Image file not found (tried 4.1.04.tiff and lena.png).")

    height, width, _ = host_image.shape
    if height % block_size != 0 or width % block_size != 0:
        print("Trimming image to be divisible by 8...")
        height = (height // block_size) * block_size
        width = (width // block_size) * block_size
        host_image = host_image[:height, :width]

    num_blocks_per_channel = (height // block_size) * (width // block_size)
    total_channel_blocks = num_blocks_per_channel * 3
    print(f"Image Size: {width}x{height}. Total Blocks: {total_channel_blocks}")

    channels = cv2.split(host_image) # [Blue, Green, Red]

    # 2. Prepare Matrices
    H_d = hadamard(d)
    W_d = hadamard(d)
    M_d = np.bitwise_and(np.where(H_d > 0, 1, 0), np.where(W_d > 0, 1, 0))

    # 3. Generate Data Pools
    msb_blocks_all = [[], [], []]
    t1_values_all = [[], [], []]
    recovery_data_all = [[], [], []]

    print("Generating Authentication and Recovery Data...")
    for ch_id in range(3):
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = channels[ch_id][i:i+block_size, j:j+block_size]
                
                # Zero out LSB to get pure Content
                msb_block = np.bitwise_and(block, 254)
                msb_blocks_all[ch_id].append(msb_block)

                # Calc T1
                S_d = (M_d @ msb_block @ M_d) / d
                T1 = (int(np.trace(S_d)) + 1) % (2**10)
                t1_values_all[ch_id].append(T1)

                # Calc Recovery Bits
                rec_bits = create_recovery_bits(msb_block)
                recovery_data_all[ch_id].append(rec_bits)

    # 4. Calculate T2 (Group checksums)
    t2_values_all = [[], [], []]
    for ch_id in range(3):
        for i in range(0, num_blocks_per_channel, 5):
            group = t1_values_all[ch_id][i:i+5]
            if len(group) == 5:
                T2 = int((sum(group) / 5) % (2**6))
                t2_values_all[ch_id].append(T2)

    # 5. Generate Dual Mappings
    # We treat all blocks from all channels as one giant pool to mix colors
    # Map 1: Primary Backup
    P_inv_1 = create_block_mapping(total_channel_blocks, SECRET_KEY_1)[1]
    # Map 2: Secondary Backup
    P_inv_2 = create_block_mapping(total_channel_blocks, SECRET_KEY_2)[1]

    print("Embedding Data (Auth + Backup 1 + Backup 2)...")
    
    watermarked_channels = [np.zeros_like(c) for c in channels]
    t2_counters = [0, 0, 0]

    for total_idx in range(total_channel_blocks):
        ch_id = total_idx // num_blocks_per_channel
        blk_idx = total_idx % num_blocks_per_channel
        
        # Current Block Data
        B_msb = msb_blocks_all[ch_id][blk_idx]
        T1 = t1_values_all[ch_id][blk_idx]
        
        if blk_idx > 0 and blk_idx % 5 == 0:
            t2_counters[ch_id] += 1
        curr_t2_idx = t2_counters[ch_id]
        T2 = t2_values_all[ch_id][curr_t2_idx] if curr_t2_idx < len(t2_values_all[ch_id]) else 0

        # Retrieve Recovery Data for OTHER blocks that are stored HERE
        # Who mapped to me in Map 1?
        source_idx_1 = P_inv_1[total_idx]
        src_ch_1 = source_idx_1 // num_blocks_per_channel
        src_blk_1 = source_idx_1 % num_blocks_per_channel
        recovery_bits_1 = recovery_data_all[src_ch_1][src_blk_1]

        # Who mapped to me in Map 2?
        source_idx_2 = P_inv_2[total_idx]
        src_ch_2 = source_idx_2 // num_blocks_per_channel
        src_blk_2 = source_idx_2 % num_blocks_per_channel
        recovery_bits_2 = recovery_data_all[src_ch_2][src_blk_2]

        # Construct Payload (64 bits)
        # Format: [T1(10)] + [T2(6)] + [Rec1(24)] + [Rec2(24)]
        payload_str = f"{T1:010b}{T2:06b}"
        payload_str += "".join(str(b) for b in recovery_bits_1)
        payload_str += "".join(str(b) for b in recovery_bits_2)
        payload_bits = [int(b) for b in payload_str]

        # Chaotic Scrambling of positions
        mean_val = np.mean(B_msb)
        std_val = np.std(B_msb)
        p1 = (mean_val + 1) / 257.0
        p2 = min(4.0, 3.99 + (std_val - int(std_val)) * 0.1)
        
        perm_indices = generate_chaotic_permutation(p1, p2, length=64)
        
        # Embed into LSB
        flat = B_msb.flatten()
        for k in range(64):
            pixel_idx = perm_indices[k]
            flat[pixel_idx] += payload_bits[k] # Add bit to LSB (MSB was cleared)

        # Reconstruct
        row = blk_idx // (width // block_size)
        col = blk_idx % (width // block_size)
        y, x = row * block_size, col * block_size
        watermarked_channels[ch_id][y:y+block_size, x:x+block_size] = flat.reshape(8,8)

    final_img = cv2.merge(watermarked_channels)
    cv2.imwrite('lena_watermarked_with_recovery_color.png', final_img)
    print("Success: 'lena_watermarked_with_recovery_color.png' created.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()