from game24_prompt import propose_prompt, value_prompt
from openai import OpenAI
# from dotenv import load_dotenv
# load_dotenv()
client = OpenAI()

def chatgpt(prompt, model="gpt-4.1-mini", n=1) -> list:
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        for _ in range(cnt):
            res = client.responses.create(model=model, input=prompt)
            print('res:', res)
            # 根据 responses 接口结构提取文本输出
            try:
                text = res.output[0].content[0].text
            except Exception:
                text = str(res)
            outputs.append(text)
            print('outputs:', outputs)
    return outputs

def first_think(input):
    proposals = chatgpt(propose_prompt.format(input=input))[0].split('\n')
    # print(proposals)
    # # 每个元素后加上\n （为了拼接后续步骤）
    # # [2 + 8 = 10 (left: 8 10 14)\n,8 / 2 = 4 (left: 4 8 14)\n]
    proposals = [_ + '\n' for _ in proposals]
    print('proposals:',proposals)
    # # # 创建对应的index list
    ids = list(range(len(proposals)))
    print('ids:',ids)

    return (ids,proposals)

def first_evaluate(proposals):
    values = []
    for proposal in proposals:
        current_numbers = proposal.strip().split('\n')[-1].split('left: ')[-1].split(')')[0]
        print('current_numbers:',current_numbers)
        if current_numbers in value_cache:
            values.append(0)
            continue
        value_outputs = chatgpt(value_prompt.format(input=current_numbers), n=3)
        print(value_outputs,'value_outputs')
        value_cache.append(current_numbers)
        """从三次的结果中取出分值：0.003"""
        value = 0
        try:
            value = sum([float(_.split('\n')[-1].split('left: ')[-1]) for _ in value_outputs])
        except:
            pass
        values.append(value)
        # 日志
        temp = proposal.replace("\n", "\\n")
        print(f"[{temp}] 得分: {value}")
    return values

def first_screen(ids,values):
    # 根据预测结果选择前5个最有可能到达24的候选项
    # 按照评级排序下一步动作的index 并记录对应的下一步动作
    select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[:5]
    select_proposals = [proposals[select_id] for select_id in select_ids]

    return select_proposals

def second_think(select_proposals):
    # 获得可能的下一步动作
    proposals = []
    for proposal in select_proposals:
        # 从第一轮候选结果中取出数组 获取下一步
        # 2 * 3 = 6 (left: 2 2 6)\n -》 2 2 6
        current_numbers = proposal.strip().split('\n')[-1].split('left: ')[-1].split(')')[0]
        # 2 + 2 = 4 (left: 4 6)\n
        # 2 * 2 = 4 (left: 4 6)\n
        # 6 - 2 = 4 (left: 2 4)\n
        p = chatgpt(propose_prompt.format(input=current_numbers))[0].split('\n')
        # 将第一步的结果和本次拼接
        # ['2 * 3 = 6 (left: 2 2 6)\n2 + 2 = 4 (left: 4 6)\n','2 * 3 = 6 (left: 2 2 6)\n6 - 2 = 4 (left: 2 4)\n']
        for item in p:
            proposals.append(proposal + item + '\n')

    ids = list(range(len(proposals)))
    return (ids,proposals)

def second_evaluate(proposals):
    values = []
    for proposal in proposals:
        current_numbers = proposal.strip().split('\n')[-1].split('left: ')[-1].split(')')[0]
        if current_numbers in value_cache:
            # print(f"{current_numbers}已评估")
            values.append(0)
            continue
        # print(f"评估:{current_numbers}...")
        value_outputs = chatgpt(value_prompt.format(input=current_numbers), n=3)
        print(value_outputs, 'value_outputs')
        value_cache.append(current_numbers)
        value = 0
        try:
            value = sum([float(_.split('\n')[-1].split('left: ')[-1]) for _ in value_outputs])
        except:
            pass
        values.append(value)
        # 日志
        temp = proposal.replace("\n", "\\n")
        print(f"[{temp}] 得分: {value}")
    return values

def second_screen(values):
    select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[:5]
    select_proposals = [proposals[select_id] for select_id in select_ids]

    return select_proposals

def third_think(select_proposals):
    # 获得可能的下一步动作
    proposals = []
    for proposal in select_proposals:
        current_numbers = proposal.strip().split('\n')[-1].split('left: ')[-1].split(')')[0]
        p = chatgpt(propose_prompt.format(input=current_numbers))[0].split('\n')
        for item in p:
            proposals.append(proposal + item + '\n')


    return proposals

def third_evaluate(proposals):
    result = []
    for proposal in proposals:
        current_numbers = proposal.strip().split('\n')[-1].split('left: ')[-1].split(')')[0]
        if current_numbers == '24':
            result.append(proposal)
    return result

def GetResult(result):
    print(f"输入的数字为：{input}。")
    if len(result) == 0:
        print("无法获得24")
    else:
        print("获得24的算式有:")
        for r in result:
            print("=================")
            print(r)

# 程序主入口
if __name__ == '__main__':
    value_cache= []
    input = "5 8 11 13"

    print("第一轮：思考生成中......")
    ids,proposals = first_think(input)
    print("第一轮思考完毕，结果：",ids,proposals)

    print("第一轮：评估结果中......")
    values = first_evaluate(proposals)
    print("第一轮评估完毕，结果：", values)

    print("第一轮：筛选候选项......")
    select_proposals = first_screen(ids,values)
    print("第一轮候选项筛选完毕，结果：", select_proposals)

    print("第二轮：思考生成中......")
    ids,proposals = second_think(select_proposals)
    print('第二轮思考完毕，结果：',proposals)

    print("第二轮：评估结果中......")
    values = second_evaluate(proposals)
    print("第二轮评估完毕，结果：", values)

    print("第二轮：筛选候选项中......")
    select_proposals = second_screen(values)
    print("第二轮候选项筛选完毕，结果：", select_proposals)

    print("第三轮：思考生成中......")
    proposals= third_think(select_proposals)
    print("第三轮思考完毕，结果：",proposals)


    print("第三轮：评估结果中......")
    result = third_evaluate(proposals)
    print("第三轮评估完毕，结果：", result)

    print('获取最终结果结果......')
    GetResult(result)






