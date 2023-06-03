import utils

if __name__ == '__main__':
    # create_data_lists_COCO(COCO_path='./COCO',
    #                     output_folder='./output')
    utils.create_data_lists_COCO_new(COCO_path='./filtered_coco_dataset_2017/',
                        output_folder='./output')
