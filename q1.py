import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageSegmenter:
    def __init__(self, image_path):
        """Initialize the ImageSegmenter with an input image."""
        self.original = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        self.segmentation_result = None

    def check_boundaries(self, point):
        """Check if a given point is within the image boundaries."""
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height

    def segment_image(self, seed_coordinates, similarity_threshold=180):
        """Perform region growing segmentation using a list for pixel processing."""
        result_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        to_process = [(seed_coordinates, self.image[seed_coordinates[1], seed_coordinates[0]])]
        result_mask[seed_coordinates[1], seed_coordinates[0]] = 255

        # Initialize region properties
        region_sum = self.image[seed_coordinates[1], seed_coordinates[0]].copy().astype(np.float32)
        pixels_in_region = 1

        # Define neighborhood connectivity (4-connectivity)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while to_process:
            (current_x, current_y), _ = to_process.pop(0)

            # Check neighboring pixels
            for dx, dy in directions:
                next_x, next_y = current_x + dx, current_y + dy

                if not self.check_boundaries((next_x, next_y)) or result_mask[next_y, next_x] != 0:
                    continue

                # Compute color similarity
                current_mean = region_sum / pixels_in_region
                color_difference = np.linalg.norm(self.image[next_y, next_x] - current_mean)

                # Add similar pixels to the region
                if color_difference < similarity_threshold:
                    result_mask[next_y, next_x] = 255
                    to_process.append(((next_x, next_y), self.image[next_y, next_x]))
                    region_sum += self.image[next_y, next_x]
                    pixels_in_region += 1

        self.segmentation_result = result_mask
        return result_mask

    def visualize_results(self, seed_point):
        """Display the segmentation results."""
        if self.segmentation_result is None:
            raise ValueError("Run segment_image first!")

        # Create visualization images
        marked_original = self.image.copy()
        cv2.circle(marked_original, seed_point, 5, (0, 0, 255), -1)  # Mark the seed point in red

        masked_result = cv2.bitwise_and(self.image, self.image, mask=self.segmentation_result)

        # Display results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.imshow(marked_original)
        ax1.set_title("Original Image with Seed Point")
        ax1.axis("off")

        ax2.imshow(masked_result)
        ax2.set_title("Segmented Region")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

    def visualize_multiple_seeds(self, seed_sets):
        """Display segmentation results for multiple sets of seed points."""
        num_sets = len(seed_sets)
        fig, axes = plt.subplots(num_sets, 2, figsize=(15, 5*num_sets))
        
        for i, seed_set in enumerate(seed_sets):
            # Create a combined mask for all seeds in this set
            combined_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            marked_original = self.image.copy()
            
            # Process each seed point in the set
            for seed_point in seed_set:
                # Mark seed point on original image
                cv2.circle(marked_original, seed_point, 5, (0, 0, 255), -1)
                
                # Segment for this seed point
                mask = self.segment_image(seed_point)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Create masked result
            masked_result = cv2.bitwise_and(self.image, self.image, mask=combined_mask)
            
            # Display results
            axes[i, 0].imshow(marked_original)
            axes[i, 0].set_title(f"Set {i+1}: Original Image with Seed Points")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(masked_result)
            axes[i, 1].set_title(f"Set {i+1}: Segmented Region")
            axes[i, 1].axis("off")
        
        plt.tight_layout()
        plt.show()

def main():
    # Configuration
    image_path = "Lion.jpg"
    seed_sets = [
        [(50, 100), (150, 150), (275, 250)],
        [(450, 190), (670, 670), (600, 150)],
        [(850, 250), (750, 450), (600, 440)]
    ]
    threshold = 180

    # Process image
    segmenter = ImageSegmenter(image_path)
    segmenter.visualize_multiple_seeds(seed_sets)

if __name__ == "__main__":
    main()
