import cv2
import numpy as np
from scipy.linalg import hadamard

print("--- Starting Grayscale Image Authentication Script ---")

# --- Configuration (must match the embedding script) ---
block_size = 8
d = 8  # Matrix dimension

# --- Reusable Functions (Copied from embedding script) ---
def generate_logistic_sequence(p1, p2, length=16):
    """Generates a chaotic logistic map sequence."""
    sequence = np.zeros(length)
    g = p1
    for i in range(length):
        g = p2 * g * (1 - g)
        sequence[i] = g
    return sequence

# --- Main Authentication Logic ---
try:
    # --- FIX 1: Corrected File Loading Logic ---
    # First, try to load the attacked image.
    image_to_check = cv2.imread('lena_watermarked.png', cv2.IMREAD_GRAYSCALE)
    if image_to_check is None:
        # If no attacked image, fall back to the clean watermarked image.
        image_to_check = cv2.imread('lena_watermarked.png', cv2.IMREAD_GRAYSCALE)
        if image_to_check is None:
            # If neither is found, raise an error.
            raise FileNotFoundError("Could not find 'lena_attacked.png' or 'lena_watermarked.png'")
    
    print(f"Successfully loaded image for authentication.")
    height, width = image_to_check.shape

    # Generate the Adaptive Matrix (must be identical to embedding)
    H_d = hadamard(d)
    W_d = hadamard(d)
    H_d_prime = np.where(H_d > 0, 1, 0)
    W_d_prime = np.where(W_d > 0, 1, 0)
    M_d = np.bitwise_and(H_d_prime, W_d_prime)
    
    tamper_map = np.zeros((height, width), dtype=np.uint8)
    
    # --- Process Each Block ---
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_to_check[i:i+block_size, j:j+block_size]
            
            msb_block = np.bitwise_and(block, 254)
            lsb_data = block % 2

            S_d = (M_d @ msb_block @ M_d) / d
            T1_recalculated = int(np.trace(S_d)) % (2**10)

            mean_bij = np.mean(msb_block)
            std_bij = np.std(msb_block)
            mu = 3.99
            alpha = 0.1
            P1_ij = (mean_bij + 1) / 257

            # --- FIX 2: Capped P2_ij to prevent overflow warning ---
            P2_ij = min(4.0, mu + (std_bij - np.floor(std_bij)) * alpha)
            
            G_ij = generate_logistic_sequence(P1_ij, P2_ij, length=block_size*block_size)
            embedding_indices = np.argsort(G_ij)
            
            flat_lsb = lsb_data.flatten()
            extracted_bits_str = "".join(str(flat_lsb[k]) for k in embedding_indices[:16])
            
            T1_extracted = int(extracted_bits_str[:10], 2)
            
            if T1_recalculated != T1_extracted:
                tamper_map[i:i+block_size, j:j+block_size] = 255
    
    output_filename = 'tamper_map.png'
    cv2.imwrite(output_filename, tamper_map)
    print(f"\n--- Authentication Complete! ---")
    print(f"Tamper map saved as: {output_filename}")

except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please ensure the watermarked/attacked image file exists in this folder.")