from PIL import Image, ImageFilter

def normalizeImage(imagePath):
    #Opening the Image
    img = Image.open(imagePath)
    #img.show()
    #Converting from png to RGB
    #img = img.convert('RGB')
    #No point in conversion to RGB as this are grayscaled image (149,149,149)

    #get the dimension of the Image
    #print(img.getbbox())

    #get band names ('R','G','B')
    #print(img.getbands())

    #Resizing the Image into lower pixels
    img = img.resize((64,64))
    #img.show()


    #Filtering the Image
    #https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    #img.show()

    #Thresholding the Image
    img = img.convert("1", None, None)
    #img.show()

    #Fetching image data in flattened way
    pixels = img.getdata()
    #print(list(pixels))

    #Convert to NP array and normalize the values
    normalized_data = np.array(list(pixels))//255
    #normalized_data = np.append(normalized_data,"Cardiomegaly")
    #print(normalized_data)
    return normalized_data

    # # By reshaping we are making it a 2d array 
    # # with 1 row and 65 columns
    # normalized_data = normalized_data.reshape(1,65)
    # print(normalized_data)

    # #Save the normalized data as csv file
    # df = pd.DataFrame(normalized_data)
    # df.to_csv('./Datasets/ChestX-RayData/Prepared-Data.csv')

    # data = pd.read_csv('./Datasets/ChestX-RayData/Prepared-Data.csv')

    # #Show the Image
    # img.show()