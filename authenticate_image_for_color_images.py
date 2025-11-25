import cv2
import numpy as np
from scipy.linalg import hadamard

print("--- Starting Color Image Authentication Script ---")

# --- Configuration (must match the embedding script) ---
block_size = 8
d = 8  # Matrix dimension

# --- Reusable Functions ---
def generate_logistic_sequence(p1, p2, length=16):
    """Generates a chaotic logistic map sequence."""
    sequence = np.zeros(length)
    g = p1
    for i in range(length):
        g = p2 * g * (1 - g)
        sequence[i] = g
    return sequence

def authenticate_channel(image_channel):
    """Applies the full authentication process to a single color channel."""
    height, width = image_channel.shape
    
    # Generate the identical Adaptive Matrix
    H_d = hadamard(d)
    W_d = hadamard(d)
    H_d_prime = np.where(H_d > 0, 1, 0)
    W_d_prime = np.where(W_d > 0, 1, 0)
    M_d = np.bitwise_and(H_d_prime, W_d_prime)
    
    channel_tamper_map = np.zeros_like(image_channel)
    
    # Process each block in the channel
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_channel[i:i+block_size, j:j+block_size]
            
            msb_block = np.bitwise_and(block, 254)
            lsb_data = block % 2

            # Recalculate T1 from MSB content
            S_d = (M_d @ msb_block @ M_d) / d
            T1_recalculated = int(np.trace(S_d)) % (2**10)

            # Extract T1 from LSB data
            mean_bij = np.mean(msb_block)
            std_bij = np.std(msb_block)
            P1_ij = (mean_bij + 1) / 257
            # Includes the fix to prevent overflow warning
            P2_ij = min(4.0, 3.99 + (std_bij - np.floor(std_bij)) * 0.1)
            
            G_ij = generate_logistic_sequence(P1_ij, P2_ij, length=block_size*block_size)
            embedding_indices = np.argsort(G_ij)
            
            flat_lsb = lsb_data.flatten()
            extracted_bits_str = "".join(str(flat_lsb[k]) for k in embedding_indices[:16])
            
            T1_extracted = int(extracted_bits_str[:10], 2)
            
            # Compare and mark tampered blocks
            if T1_recalculated != T1_extracted:
                channel_tamper_map[i:i+block_size, j:j+block_size] = 255
                
    return channel_tamper_map

# --- Main Authentication Logic ---
try:
    # Load the color watermarked image (and possibly tampered)
    image_to_check = cv2.imread('lena_attacked.png', cv2.IMREAD_COLOR)
    if image_to_check is None:
        image_to_check = cv2.imread('lena_watermarked_color.png', cv2.IMREAD_COLOR)
        if image_to_check is None:
            raise FileNotFoundError("Could not find attacked or watermarked color image.")
    
    print("Successfully loaded color image for authentication.")
    
    # Split the image into its Blue, Green, and Red channels
    blue_channel, green_channel, red_channel = cv2.split(image_to_check)
    
    # Authenticate each channel independently
    print("Authenticating Blue channel...")
    tamper_map_blue = authenticate_channel(blue_channel)
    
    print("Authenticating Green channel...")
    tamper_map_green = authenticate_channel(green_channel)
    
    print("Authenticating Red channel...")
    tamper_map_red = authenticate_channel(red_channel)
    
    # Merge the individual tamper maps into a final color map
    # Tampered areas in the red channel will appear red, blue will be blue, etc.
    # Areas tampered in all channels will appear white.
    final_tamper_map = cv2.merge([tamper_map_blue, tamper_map_green, tamper_map_red])
    
    output_filename = 'tamper_map.png'
    cv2.imwrite(output_filename, final_tamper_map)
    print(f"\n--- Authentication Complete! ---")
    print(f"Color tamper map saved as: {output_filename}")

except FileNotFoundError as e:
    print(f"ERROR: {e}")