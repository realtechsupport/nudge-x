import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

def tiff_to_png_rasterio(input_tiff, output_png):
    try:
        # Open the TIFF file
        with rasterio.open(input_tiff) as src:
            # Read all bands
            image = src.read()
            
            # Handle different numbers of bands
            if image.shape[0] == 1:  # Single band
                # Normalize to 0-255 range
                image = image[0]
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                cmap = 'gray'
            
            elif image.shape[0] == 3:  # RGB
                # Normalize and transpose to HxWxC
                image = image.transpose(1, 2, 0)
                # Normalize each channel to 0-255 range
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                cmap = None
            
            elif image.shape[0] == 4:  # RGBA
                # Normalize and transpose to HxWxC
                image = image.transpose(1, 2, 0)
                # Normalize each channel to 0-255 range
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                cmap = None
            
            # Create the plot
            plt.figure(figsize=(10, 10))
            plt.axis('off')
            
            if cmap:
                plt.imshow(image, cmap=cmap)
            else:
                plt.imshow(image)
            
            # Save as PNG
            plt.savefig(output_png, 
                       bbox_inches='tight', 
                       pad_inches=0, 
                       dpi=300,
                       format='png')
            plt.close()
            
        print(f"Successfully converted {input_tiff} to {output_png}")
        return True

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False
        
#----------------------------------------------------------------------

def tiff_to_png_rasterio_v2(input_tiff, output_png):
    try:
        # Open the TIFF file
        with rasterio.open(input_tiff) as src:
            # Read all bands
            data = src.read()
            
            # Handle different numbers of bands
            if data.shape[0] == 1:  # Single band (grayscale)
                # Scale data to 0-255 range
                data_scaled = numpy.clip(data[0], 0, 255).astype(numpy.uint8)
                img = Image.fromarray(data_scaled, mode='L')
                
            elif data.shape[0] == 3:  # Three bands (RGB)
                # Scale data to 0-255 range for each band
                data_scaled = numpy.zeros_like(data, dtype=numpy.uint8)
                for band in range(3):
                    # Linear stretch with percentile clipping
                    p2, p98 = numpy.percentile(data[band], (2, 98))
                    data_scaled[band] = numpy.clip(
                        ((data[band] - p2) * (255.0 / (p98 - p2))),
                        0, 255
                    ).astype(numpy.uint8)
                
                # Transpose to correct shape (height, width, channels)
                data_scaled = numpy.transpose(data_scaled, (1, 2, 0))
                img = Image.fromarray(data_scaled, mode='RGB')
                
            elif data.shape[0] == 4:  # Four bands (RGBA)
                # Scale data to 0-255 range for each band
                data_scaled = numpy.zeros_like(data, dtype=numpy.uint8)
                for band in range(3):  # Only adjust RGB bands
                    p2, p98 = numpy.percentile(data[band], (2, 98))
                    data_scaled[band] = numpy.clip(
                        ((data[band] - p2) * (255.0 / (p98 - p2))),
                        0, 255
                    ).astype(numpy.uint8)
                # Copy alpha band directly
                data_scaled[3] = numpy.clip(data[3], 0, 255).astype(numpy.uint8)
                
                # Transpose to correct shape (height, width, channels)
                data_scaled = numpy.transpose(data_scaled, (1, 2, 0))
                img = Image.fromarray(data_scaled, mode='RGBA')
            
            # Save the image
            img.save(output_png)
            
        print(f"Successfully converted {input_tiff} to {output_png}")
        return True

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False