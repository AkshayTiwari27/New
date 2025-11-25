import cv2
import numpy as np
from scipy.linalg import hadamard

print("--- Starting Watermark Embedding Script ---")

# --- Step 1: Load Image and Divide into Blocks ---
try:
    # Make sure your image file is in the same folder and the name matches!
    host_image = cv2.imread('4.1.04.tiff', cv2.IMREAD_GRAYSCALE)
    if host_image is None:
        raise FileNotFoundError("Image file not found or could not be opened.")
    
    print("Successfully loaded the image.")
    block_size = 8
    height, width = host_image.shape
    
    image_blocks = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = host_image[i:i+block_size, j:j+block_size]
            image_blocks.append(block)
    
    print(f"Image divided into {len(image_blocks)} blocks.")

except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please make sure your image (e.g., 'lena.tiff') is in the same folder as this script.")
    exit()


# --- Step 2: Clear the Least Significant Bit (LSB) ---
msb_blocks = []
for block in image_blocks:
    # This is equivalent to block - (block mod 2)
    msb_block = np.bitwise_and(block, 254) 
    msb_blocks.append(msb_block)
print("Cleared the LSB from all image blocks.")


# --- Step 3: Generate the Adaptive Matrix (M_d) ---
d = 8  # Our block size
H_d = hadamard(d)
W_d = hadamard(d) # As per our simplified approach

# Remove negative values (element-wise thresholding)
H_d_prime = np.where(H_d > 0, 1, 0)
W_d_prime = np.where(W_d > 0, 1, 0)

# Generate the final adaptive matrix with a logical AND
M_d = np.bitwise_and(H_d_prime, W_d_prime)
print("Successfully generated the adaptive matrix M_d.")

print("\n--- Initial Setup Complete! Ready for the next steps. ---")

print("\n--- Initial Setup Complete! Ready for the next steps. ---")


print("\n--- Starting Steps 4 & 5: Calculating Validation Matrix T1 ---")

T1_values = []
for B_ij in msb_blocks:
    # Step 4: Compute the unsigned matrix S_d for each block [cite: 208]
    S_d = (M_d @ B_ij @ M_d) / d
    
    # Step 5: Sum the diagonal of S_d and map to a 10-bit range (0-1023) for T1 [cite: 213, 178]
    T1_ij = int(np.trace(S_d)) % (2**10) # np.trace() sums the diagonal
    T1_values.append(T1_ij)

print(f"Calculated {len(T1_values)} values for the first validation matrix (T1).")


print("\n--- Starting Step 6: Calculating Validation Matrix T2 ---")

# The paper groups blocks into sets of 5 to calculate T2 [cite: 182]
T2_values = []
num_blocks = len(T1_values)
for i in range(0, num_blocks, 5):
    # Get the T1 values for the current group of up to 5 blocks
    group_of_T1s = T1_values[i:i+5]
    
    # The paper's formula implies using a full group of 5
    if len(group_of_T1s) == 5:
        # Compute the second validation value T2_pq (6 bits, 0-63) 
        sum_of_T1s = sum(group_of_T1s)
        T2_pq = int( (sum_of_T1s / 5) % (2**6) )
        T2_values.append(T2_pq)

# The number of T2 values will be 1/5th the number of blocks
print(f"Calculated {len(T2_values)} values for the second validation matrix (T2).")

print("\n--- Validation Data Generated! Ready for the next step: Embedding. ---")

# --- Step 7: Generate Logistic Sequence & Embed Watermark ---
print("\n--- Starting Final Step: Embedding Watermark and Saving Image ---")

def generate_logistic_sequence(p1, p2, length=16):
    """Generates a chaotic logistic map sequence."""
    sequence = np.zeros(length)
    g = p1
    for i in range(length):
        g = p2 * g * (1 - g)
        sequence[i] = g
    return sequence

watermarked_blocks = []
block_index = 0
t2_index = 0

for B_ij in msb_blocks:
    # --- Part A: Prepare the 16 watermark bits ---
    if block_index % 5 == 0 and block_index > 0:
        t2_index += 1 # Move to the next T2 value for the new group of 5
    
    current_t1 = T1_values[block_index]
    # Ensure we don't go out of bounds for T2
    current_t2 = T2_values[t2_index] if t2_index < len(T2_values) else 0

    # Convert T1 (10 bits) and T2 (6 bits) to binary strings
    t1_bits = format(current_t1, '010b')
    t2_bits = format(current_t2, '06b')
    watermark_bits_str = t1_bits + t2_bits
    watermark_bits = [int(bit) for bit in watermark_bits_str]

    # --- Part B: Generate the chaotic sequence for the current block ---
    mean_bij = np.mean(B_ij)
    std_bij = np.std(B_ij)
    mu = 3.99 # A value for strong chaos
    alpha = 0.1

    P1_ij = (mean_bij + 1) / (2**8 + 1)
    P2_ij = min(4.0, mu + (std_bij - np.floor(std_bij)) * alpha)
    
    # Generate the chaotic sequence G
    G_ij = generate_logistic_sequence(P1_ij, P2_ij, length=block_size*block_size)
    
    # --- Part C: Embed the 16 bits into the block ---
    # Get a pseudo-random permutation of pixel indices from the chaotic sequence
    embedding_indices = np.argsort(G_ij)
    
    # Flatten the block to easily access pixels by index
    flat_block = B_ij.flatten()
    
    # Embed the 16 watermark bits into the first 16 pixel locations from our permutation
    for i in range(16):
        pixel_index = embedding_indices[i]
        # Add the watermark bit (0 or 1). Since LSB is 0, this is the same as setting it.
        flat_block[pixel_index] += watermark_bits[i]
        
    # Reshape the block back to its original 8x8 size and add to our list
    watermarked_block = flat_block.reshape((block_size, block_size))
    watermarked_blocks.append(watermarked_block)
    
    block_index += 1

print("Successfully embedded watermark data into all blocks.")


# --- Step 8: Reconstruct and Save the Final Image ---
# Reconstruct the image from the watermarked blocks
reconstructed_image = np.zeros((height, width), dtype=np.uint8)
block_idx = 0
for i in range(0, height, block_size):
    for j in range(0, width, block_size):
        reconstructed_image[i:i+block_size, j:j+block_size] = watermarked_blocks[block_idx]
        block_idx += 1

# Save the final watermarked image
output_filename = 'lena_watermarked.png'
cv2.imwrite(output_filename, reconstructed_image)

print(f"\n--- Embedding Process Complete! ---")
print(f"Watermarked image saved as: {output_filename}")




