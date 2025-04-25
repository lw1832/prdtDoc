"""
方案一：
A (读取pdf, 输入：文件地址 file_path): 获取pdf中各种文字、图片 输出 dict
B (识别pdf图片语义, 输入 pdf-dict): 为图片打标  输出 pdf-dict | 图片-dict(k-v)

C (将打标图片、pdf文本 (dict)、prompt (str)作为输入): 输出 按照模板和要求总结的产品推荐数据结构对象 AIMessage

D (将C中内容作为输入 AIMessage): 将数据结构对象填入模板，导出对应文档
"""
import base64
import io
import json
import os
from typing import TypedDict, Literal, Optional, List

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from openai import BaseModel

from generate_para import genertate_output_md
from scripts.ocr_layout import process


# 定义State
class PDState(TypedDict):
    file_path: str # 文件地址
    file_content: dict # 文件解析结果
    ai_result: BaseMessage # 生成的产品推荐手册文本信息数据结构
    save_dir: str # 保存路径


# 定义段落模块数据结构
class PeriodInstructions(TypedDict):
    """
     Instruction of representing paragraph content. Image identifier when type='img'
     Attributes:
        type (Literal['txt', 'img']): Content type discriminator
            - 'txt': Indicates a text paragraph
            - 'img': Indicates an image paragraph
        content (str): The actual content payload
            - For type='txt': Contains the text string
            - For type='img': Contains the image resource identifier

    Examples:
        - text_block = Paragraph(type='txt', content='Lorem ipsum')
        - image_block = Paragraph(type='img', content='fig_001.png')
    """
    type: Literal['txt', 'img']
    content: str


class ImageDescriptionInstructions(BaseModel):
    """
    Instruction of image-description task output.
    该数据结构用于图片文本内容描述任务的输出模板，其中包含两个字段，type代表该图片类型的推理，如果推测该图片是ppt的符号图形，则为'ppt'，否则为'img'
    description存放任务输出的图片文本内容描述信息
    """
    type: Literal['ppt', 'img']
    description: str

# 定义生成产品推荐手册文本对象数据结构
class PrdtDocInstructions(BaseModel):
    """
    Instruction of Product recommendation manual document content.
    Each member variable is implemented as a list, where the list order determines the final paragraph sequence in the
    generated document.
    """
    general_introduction: list[PeriodInstructions]
    core_functions_introduction: list[PeriodInstructions]
    case_analysis: list[PeriodInstructions]
    docking_process: list[PeriodInstructions]


class PrdtDocInstruct(TypedDict):
    type: Literal['title', 'txt', 'img']
    content: str
    sub_obj: Optional[List['PrdtDocInstruct']]


# 输入有两个数组，第一个数组传入了供你生成推荐手册文档的文字材料；第二个数组传入了图片材料，每个图片数据有两个属性，其中name表示其编号，image表示该图片的信息
os.environ["OPENAI_API_KEY"] = "sk-LGylSerFax4P4OiNZAKHyLQnf0VLHAcpltDsy4OehkPLXkUC"
os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"
prompt = """
    你是一个邮政储蓄银行的产品推荐手册文档的写作专家，请阅读、润色给你的材料，并以下述的模块生成一个完整的产品推荐手册文档：
    总体介绍、核心功能介绍、案例分析、对接流程
    说明：
    1. 输入为json格式，里面包含了一段文本材料，和若干图片材料，我不会直接上传图片材料，而是给你提供了图片的名称(全路径名)、图片文字描述以及标签，请理解这些材料的内容
    2. 输出必须包含这四个模块
    输出内容要求：
    1. “总体介绍”部分的内容需包括但不限于建设背景、业务定位及应用场景、客户服务群体、主要功能清单/功能分类等。在语言表述方面，弱化技术架构、技术语言等描述，从功能优势、服务场景等有助于业务发展方面展开描述。
    2. “核心功能介绍”部分主要阐述解决客户痛点、带来的业务价值。内容包括但不限于该产品主要功能的服务场景、业务价值、能够解决的客户的痛点及问题、与同业对比等。
    3. “案例分析”部分主要阐述该产品的功能为什么样的客群解决什么样的问题，带来什么成效。包括但不限于分行面临的痛点及问题、解决方案、给分行带来的价值和成效等。
    4. “对接流程”主要是阐述产品对接的流程、方式、内容。
    5. 每个模块尽量详细、丰富，字数控制在200-250以内，至少3个子模块（如总体介绍含背景/定位/场景/功能），能涵盖给定材料（文本+图片）中98%的内容。
    6. 要求图文并茂，请推理图片和文本的关联性，请用图片描述来补充文本，请结合文本来理解图像描述的上下文。
    7. 在理解图片材料时，应注意给出的标签(label)，此字段如果有值，则表示该图片所属的模块或者内容方向，尽量将图片材料放入输出数组中的合适位置，如果找不到图片对应的描述文字，请结合上下文生成自然段用于解释该图片。
    输出格式要求：
    1. 每个自然段如果是单独一段，则起一个小段的标题，放在文本前，用“****”包围，例如：**小段标题**正文内容。
    2. 输入的材料中还有图片数组，请识别数组中各个图片语义，根据语义和上下文，将图片穿插在文本段落的合适位置(尽量充分利用给出的图片材料，但不要有连续多张图排列一起的情况)
    3. 根据提供的工具函数接口，识别生成文本的格式，输出一个PrdtDocInstructions对象，每个模块字段中应该存放一个数组，数组中每个元素为一个PeriodInstructions对象，数组元素的顺序意味着在文章中的顺序。
    4. 如果是图片，则根据给定的PeriodInstructions数据结构生成对应代表该图片的对象，其中content应该放该图片的名称(全路径名)。
    
"""

# 定义工具集合
tools1 = [PrdtDocInstructions, PeriodInstructions]
tools2 = [ImageDescriptionInstructions]
# 定义agent
model_gen = ChatOpenAI(model='gpt-4o', temperature=0.3).bind_tools(tools1)
model_img = ChatOpenAI(model='gpt-4o', temperature=0.3).bind_tools(tools2)




# 定义agent_node
def agent_node(state: PDState) -> PDState:
    contents = state['file_content']['contents']
    figures = state['file_content']['figures']
    figures_fin = []
    for figure in figures:
        message = [
            SystemMessage(content="""
            这张图片来自邮政储蓄银行的“产品功能介绍”ppt中的其中一张截图，其中有一些图是有具体业务背景含义的图片，有一些是被目标价测算法误识别出来的ppt中的装饰图或者模板图形，
            这些ppt图形往往图片仅包含基础几何形状（圆形/扇形/箭头）或者单一图标；文字内容简短抽象（如"核心定位""01"）；与图形结合松散，缺乏数据展示；无明确信息层级结构或逻辑关系。
            请根据图片内容及其中的文本分析该图的含义。同时请根据文本内容、图标元素、业务逻辑完整性等因素关联去推测这张图片是否为本身没有具体业务指向的ppt的装饰图或者模板图形，
            如果是，则将这个图片标记为deprecated。
            
            结果输出要求：
            必须以输出ImageCaptionInstructions数据结构的形式，判断如果该图片是ppt模板图片，则type=‘ppt’否则type='img'，将该图含义信息存放到description中
            """),
            HumanMessage(content=[
                {"type": "text", "text": "请输出图片内容描述以及是否为ppt模板图片推测结果"},
                figure[1]
            ])]
        img_result = model_img.invoke(message)
        if len(img_result.tool_calls) >= 1:
            imgDscp = img_result.tool_calls[0]['args']
            print("figure:", figure[0], " type:", imgDscp['type'], " description:", imgDscp['description'])
            if imgDscp['type'] == 'img':
                figures_fin.append({'name': figure[0], 'description': imgDscp['description'], 'label': figure[2]})
        else:
            print("figure:", figure[0], " type:", img_result)
            figures_fin.append({'name': figure[0], 'description': img_result.content})

    content = '\n'.join([element for sublist in contents for element in sublist])
    message = {'文本材料': content, '图片材料': figures_fin}
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=json.dumps(message)),
    ]
    state['ai_result'] = model_gen.invoke(messages)
    return state


# 定义parse_pdf_node
def parse_pdf(state: PDState) -> PDState:
    contents, figures = process(state['file_path'])
    figures = [img_conv(i) for figure in figures for i in figure]
    state['file_content'] = {'contents': contents, 'figures': figures}
    return state

def img_conv(figure):
    image = figure['image']
    path_dir = figure['name']
    label = figure['label']

    # 将图像保存为字节流
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')  # 可以选择不同的格式，例如'JPEG'或'PNG'
    byte_data = byte_arr.getvalue()

    # 将字节数据编码为Base64
    image_data = base64.b64encode(byte_data).decode('utf-8')
    return [path_dir, {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
    }, label]


# 定义generation_node
def generation_doc(state: PDState) -> PDState:
    tool_call_results = state['ai_result'].tool_calls
    content = {}
    for tool_call_res in tool_call_results:
        content.update(tool_call_res['args'])
    file_name = os.path.splitext(os.path.basename(state['file_path']))[0]
    if not os.path.exists(state['save_dir']):
        os.makedirs(state['save_dir'])
    genertate_output_md('template.txt', os.path.join(state['save_dir'], f"{file_name}.md"), content)
    return state


# 图定义
workflow = StateGraph(PDState)
workflow.add_node("parse_pdf_node", parse_pdf)
workflow.add_node("agent_node", agent_node)
workflow.add_node("generation_node", generation_doc)
workflow.set_entry_point("parse_pdf_node")
workflow.add_edge("parse_pdf_node", "agent_node")
workflow.add_edge("agent_node", "generation_node")
workflow.set_finish_point("generation_node")


graph = workflow.compile()

input_obj = {'file_path': f"D:\\pyWorkSpace\\test\\PDF-Extract-Kit-main\\inputs\\ocr\\企业知识管理系统产品推介手册-20250423.pdf",
             'save_dir':f"D:\\pyWorkSpace\\test\\PDF-Extract-Kit-main\\scripts"}
result = graph.invoke(input_obj)

