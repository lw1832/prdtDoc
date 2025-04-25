import os
import os.path as osp
import sys

from PIL import ImageDraw

from pdf_extract_kit.utils.data_preprocess import load_pdf
import pdf_extract_kit.tasks # 勿动！！！！
sys.path.append(osp.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models
from func import ocr_layout_merge

TASK_NAME_OCR = 'ocr'
TASK_NAME_LAYOUT = 'layout_detection'
CONFIG_PATH = '../configs/ocr_layout.yaml'

def process(file_path=None):
    config = load_config(CONFIG_PATH)
    task_instances = initialize_tasks_and_models(config)

    # get input and output path from config
    input_data = file_path if file_path is not None else config.get('inputs', None)
    file_name = os.path.splitext(os.path.basename(input_data))[0]
    result_path_ocr = config.get('outputs', 'outputs'+'/'+'OCR_LAYOUT'+'/'+TASK_NAME_OCR) + '/' + file_name
    result_path_layout = config.get('outputs', 'outputs'+'OCR_LAYOUT'+'/'+TASK_NAME_LAYOUT) + '/' + file_name
    result_path_img = config.get('outputs', 'outputs'+'/img') + '/' + file_name
    visualize = config['tasks'][TASK_NAME_OCR]['visualize']

    # formula_detection_task
    task_ocr = task_instances[TASK_NAME_OCR]
    task_layout = task_instances[TASK_NAME_LAYOUT]
    print("================开始执行ocr任务================")
    ocr_results = task_ocr.process(input_data, save_dir=result_path_ocr, visualize=visualize)
    print("================ocr任务执行结束================")

    print("================开始执行layout_detection任务================")
    layout_results = task_layout.predict_pdfs(input_data, result_path_layout)
    print("================layout_detection任务执行结束================")

    print(f'Task done, results can be found at {config.get('outputs')}')
    return ocr_layout_extract(input_data, ocr_results[0], layout_results, result_path_img)

def ocr_layout_extract(fpath, ocr_results, layout_results, save_dir=""):
    images = load_pdf(fpath)
    basename = os.path.basename(fpath)[:-4]
    figures = []
    contents = []
    for idx, (image, ocr_result, layout_result) in enumerate(zip(images, ocr_results, layout_results)):
        ol_results = ocr_layout_merge(ocr_result, layout_result.__dict__['boxes'].xyxy, layout_result.__dict__['boxes'].cls, layout_result.__dict__['boxes'].conf)

        if len(ol_results['image_boxes']):
            figure_temp = []
            for i, image_box in enumerate(ol_results['image_boxes']):
                file_name = os.path.join(save_dir, f"{basename}-page{idx}-no{i}.png")
                img_cut = image.crop(image_box.xyxy)
                figure_temp.append({"image": img_cut, "name": file_name, 'label': image_box.text})
                img_cut.save(file_name)
            figures.append(figure_temp)
        if len(ol_results['text_boxes']):
            contents.append([box.text for box in ol_results['text_boxes']])
    return contents, figures

def visualize_ocr_results(fpath, ocr_results, save_dir=""):
    def visualize_image(image, ocr_res, save_path="", cate2color={}):
        """plot each result's bbox and category on image.

        Args:
            image: PIL.Image.Image
            ocr_res: list of ocr det and rec, whose format following the results of self.predict_image function
            save_path: path to save visualized image
        """
        draw = ImageDraw.Draw(image)
        for res in ocr_res:
            box_color = cate2color.get(res['category_type'], (0, 255, 0))
            x_min, y_min = int(res['bbox'][0]), int(res['bbox'][1])
            x_max, y_max = int(res['bbox'][2]), int(res['bbox'][3])
            draw.rectangle([x_min, y_min, x_max, y_max], fill=None, outline=box_color, width=1)
            draw.text((x_min, y_min), res['category_type'], (255, 0, 0))
        if save_path:
            image.save(save_path)
    images = load_pdf(fpath)
    basename = os.path.basename(fpath)[:-4]
    for idx, (ocr_result, image) in enumerate(zip(ocr_results, images)):
        visualize_image(image, ocr_result, os.path.join(save_dir, basename, f"page_cluster_{idx+1}.jpg"))

if __name__ == "__main__":
    # args = parse_args()
    # main(args.config)
    a, b = process(f"D:\\pyWorkSpace\\test\\PDF-Extract-Kit-main\\inputs\\ocr\\开放式缴费平台产品推荐手册.pdf")
    print(a, b)