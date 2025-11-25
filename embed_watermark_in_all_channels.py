import cv2
import numpy as np
from scipy.linalg import hadamard

print("--- Starting Color Watermark Embedding Script ---")

# --- Configuration ---
block_size = 8
d = 8

# --- Reusable Functions ---
def generate_logistic_sequence(p1, p2, length=16):
    sequence = np.zeros(length)
    g = p1
    for i in range(length):
        g = p2 * g * (1 - g)
        sequence[i] = g
    return sequence

def embed_watermark_on_channel(image_channel):
    """Applies the full watermarking process to a single color channel."""
    height, width = image_channel.shape
    
    image_blocks = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_channel[i:i+block_size, j:j+block_size]
            image_blocks.append(block)

    msb_blocks = [np.bitwise_and(block, 254) for block in image_blocks]

    H_d = hadamard(d)
    W_d = hadamard(d)
    H_d_prime = np.where(H_d > 0, 1, 0)
    W_d_prime = np.where(W_d > 0, 1, 0)
    M_d = np.bitwise_and(H_d_prime, W_d_prime)

    T1_values = []
    for B_ij in msb_blocks:
        S_d = (M_d @ B_ij @ M_d) / d
        T1_ij = int(np.trace(S_d)) % (2**10)
        T1_values.append(T1_ij)
    
    T2_values = []
    for i in range(0, len(T1_values), 5):
        group = T1_values[i:i+5]
        if len(group) == 5:
            T2_pq = int((sum(group) / 5) % (2**6))
            T2_values.append(T2_pq)

    watermarked_blocks = []
    block_index = 0
    t2_index = 0
    for B_ij in msb_blocks:
        if block_index > 0 and block_index % 5 == 0:
            t2_index += 1
        
        current_t1 = T1_values[block_index]
        current_t2 = T2_values[t2_index] if t2_index < len(T2_values) else 0

        watermark_bits = [int(bit) for bit in f"{current_t1:010b}{current_t2:06b}"]

        mean_bij = np.mean(B_ij)
        std_bij = np.std(B_ij)
        P1_ij = (mean_bij + 1) / 257
        
        # --- THIS IS THE FIXED LINE ---
        # We cap the value of P2_ij at 4.0 to prevent mathematical overflow.
        P2_ij = min(4.0, 3.99 + (std_bij - np.floor(std_bij)) * 0.1)
        
        G_ij = generate_logistic_sequence(P1_ij, P2_ij, length=block_size*block_size)
        embedding_indices = np.argsort(G_ij)
        
        flat_block = B_ij.flatten()
        for i in range(16):
            flat_block[embedding_indices[i]] += watermark_bits[i]
            
        watermarked_blocks.append(flat_block.reshape((block_size, block_size)))
        block_index += 1

    reconstructed_channel = np.zeros_like(image_channel)
    idx = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            reconstructed_channel[i:i+block_size, j:j+block_size] = watermarked_blocks[idx]
            idx += 1
            
    return reconstructed_channel

# --- Main Script Logic ---
try:
    host_image = cv2.imread('4.1.04.tiff', cv2.IMREAD_COLOR)
    if host_image is None:
        raise FileNotFoundError("Image file not found.")

    print("Successfully loaded color image.")
    
    blue_channel, green_channel, red_channel = cv2.split(host_image)
    
    print("Processing Blue channel...")
    watermarked_blue = embed_watermark_on_channel(blue_channel)
    
    print("Processing Green channel...")
    watermarked_green = embed_watermark_on_channel(green_channel)
    
    print("Processing Red channel...")
    watermarked_red = embed_watermark_on_channel(red_channel)
    
    watermarked_image = cv2.merge([watermarked_blue, watermarked_green, watermarked_red])
    
    output_filename = 'lena_watermarked_color.png'
    cv2.imwrite(output_filename, watermarked_image)
    
    print(f"\n--- Color Embedding Process Complete! ---")
    print(f"Color watermarked image saved as: {output_filename}")

except FileNotFoundError as e:
    print(f"ERROR: {e}")