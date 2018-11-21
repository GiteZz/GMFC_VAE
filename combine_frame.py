from PIL import Image

if __name__ == "__main__":

    frame_width = 640
    frame_height = 360
    folder = "D:/Stage/Data/Avenue/testing_videos/01_frames/"

    for i in range(1438):
        grey_image = Image.open(folder + 'greyscale/' + str(i) + '.png')
        rgb_image = Image.open(folder + 'RGB/' + str(i) + '.png')
        opt_flow_image = Image.open(folder + 'optical_flow/' + str(i) + '.png')

        new_im = Image.new('RGB', (frame_width * 2, frame_height * 2))

        new_im.paste(rgb_image, (0,0))
        new_im.paste(grey_image, (frame_width, 0))
        new_im.paste(opt_flow_image, (0, frame_height))

        new_im.save(folder + 'combined/' + str(i) + '.png')

        print(i)
