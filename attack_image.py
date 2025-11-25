import cv2
import numpy as np
import os

# --- Attack Function Definitions (These are unchanged) ---

def attack_content_removal(image, x=220, y=220, w=100, h=80):
    """(Forgery Type: Erasing) Blacks out a rectangular region of the image."""
    attacked_image = image.copy()
    cv2.rectangle(attacked_image, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return attacked_image, 'Content Removal'

def attack_copy_move(image, src_x=None, src_y=None, w=None, h=None, dst_x=None, dst_y=None):
    """(Forgery Type: Cloning) Dynamically copies a region based on image size."""
    attacked_image = image.copy()
    rows, cols, _ = attacked_image.shape

    # 1. Define region size (15% of image size if not specified)
    if w is None: w = int(cols * 0.15)
    if h is None: h = int(rows * 0.15)
    
    # 2. Define coordinates if not specified
    # Source: Bottom-Right quadrant
    if src_x is None: src_x = int(cols * 0.6)
    if src_y is None: src_y = int(rows * 0.6)
    
    # Destination: Top-Left quadrant
    if dst_x is None: dst_x = int(cols * 0.1)
    if dst_y is None: dst_y = int(rows * 0.1)

    # 3. Safety Check: Ensure coordinates stay within image bounds
    # If the calculated box goes off the edge, shift it back
    if src_x + w > cols: src_x = cols - w
    if src_y + h > rows: src_y = rows - h
    if dst_x + w > cols: dst_x = cols - w
    if dst_y + h > rows: dst_y = rows - h

    # 4. Perform the Copy-Move
    source_region = attacked_image[src_y:src_y+h, src_x:src_x+w]
    attacked_image[dst_y:dst_y+h, dst_x:dst_x+w] = source_region
    
    return attacked_image, 'Copy-Move'
def attack_political_splicing(image, face_image_path='politician_face.tiff', x=50, y=50, w=100, h=100):
    """(Forgery Type: Splicing) Pastes a face, ensuring it fits inside the image."""
    attacked_image = image.copy()
    rows, cols, _ = attacked_image.shape
    
    try:
        # 1. Load the face
        face_image = cv2.imread(face_image_path)
        if face_image is None: raise FileNotFoundError

        # 2. Safety Check: If the face is too big for the image, shrink the face
        if w >= cols or h >= rows:
            w = int(cols / 3)
            h = int(rows / 3)

        # 3. Safety Check: If coordinates go off the edge, shift them back
        if x + w > cols: x = cols - w
        if y + h > rows: y = rows - h

        # 4. Perform the splice
        face_resized = cv2.resize(face_image, (w, h))
        attacked_image[y:y+h, x:x+w] = face_resized
        return attacked_image, 'Political Splicing'

    except FileNotFoundError:
        print(f"  - ERROR: Splice image '{face_image_path}' not found.")
        return None, None

def attack_cropping(image, border=50):
    """(Forgery Type: Resizing) Crops the image, removing pixels from all sides."""
    attacked_image = image.copy()
    h, w, _ = attacked_image.shape
    cropped_image = attacked_image[border:h-border, border:w-border]
    return cv2.resize(cropped_image, (w, h)), 'Cropping'

def attack_jpeg_compression(image, quality=75):
    """(Forgery Type: File Format Manipulation) Saves and reloads the image as a lossy JPEG."""
    temp_filename = 'temp_compressed.jpg'
    cv2.imwrite(temp_filename, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    attacked_image = cv2.imread(temp_filename)
    os.remove(temp_filename)
    return attacked_image, f'JPEG Compression (Quality {quality})'

def attack_salt_and_pepper(image, amount=0.05):
    """(Forgery Type: Noise Addition) Adds random black and white pixels across the image."""
    attacked_image = image.copy()
    num_pixels = int(amount * image.size / 3)
    for _ in range(num_pixels): # Salt
        y, x = np.random.randint(0, image.shape[0]-1), np.random.randint(0, image.shape[1]-1)
        attacked_image[y, x] = (255, 255, 255)
    for _ in range(num_pixels): # Pepper
        y, x = np.random.randint(0, image.shape[0]-1), np.random.randint(0, image.shape[1]-1)
        attacked_image[y, x] = (0, 0, 0)
    return attacked_image, 'Salt and Pepper Noise'

# --- Main Interactive Script Logic ---
if __name__ == "__main__":
    try:
        source_image = cv2.imread('lena_watermarked_color.png')
        if source_image is None:
            raise FileNotFoundError("'lena_watermarked_color.png' not found.")
        
        # --- Display the Menu and Get User Input ---
        print("\n--- CHOOSE AN ATTACK TO PERFORM ---")
        print("1. Content Removal (Black Box)")
        print("2. Copy-Move Forgery")
        print("3. Political Splicing (requires 'politician_face.png')")
        print("4. Cropping")
        print("5. JPEG Compression")
        print("6. Salt and Pepper Noise")
        
        choice = input("Enter the number of the attack you want to perform: ")
        
        attacked_image = None
        attack_type = "None"

        # --- Process the User's Choice ---
        if choice == '1':
            attacked_image, attack_type = attack_content_removal(source_image)
        elif choice == '2':
            attacked_image, attack_type = attack_copy_move(source_image)
        elif choice == '3':
            attacked_image, attack_type = attack_political_splicing(source_image)
        elif choice == '4':
            attacked_image, attack_type = attack_cropping(source_image, border=30)
        elif choice == '5':
            attacked_image, attack_type = attack_jpeg_compression(source_image, quality=80)
        elif choice == '6':
            attacked_image, attack_type = attack_salt_and_pepper(source_image, amount=0.02)
        else:
            print("Invalid choice. Exiting script.")
            exit()

        # Save the attacked image if an attack was successfully performed
        if attacked_image is not None:
            output_filename = 'lena_attacked.png'
            cv2.imwrite(output_filename, attacked_image)
            print(f"\nSuccessfully performed forgery: '{attack_type}'")
            print(f"Attacked image saved as: {output_filename}")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please run the color embedding script first to create the watermarked image.")