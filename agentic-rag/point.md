数据清洗：md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')

```
Error: Error code: 400 - {'error': {'message': "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. The following tool_call_ids did not have response messages: call_ylllaIoAnJzMM6Gp8N0dTcpk", 'type': 'invalid_request_error', 'param': 'messages.[23].role', 'code': None}}
```
openai的参数检验规范：state["messages"]的AIMessage类型的信息中存在tool_call，tool_call中有id，在后续的ToolMessage类型的信息中的tool_call_id与id是一一对应的，相当于AIMessage的tool_call.id是请求，ToolMessage的tool_call_id是tool对AI的回复

这个报错的根因：上一轮对话中存在AIMessage的tool_call.id，但没有ToolMessage的tool_call_id与之对应，导致在下一轮对话时，llm校验出错，打印出以上报错信息。

逻辑原因：路由节点，当状态中自定义的循环或者工具调用次数超过自定义配置阈值时，携带tool_call.id的AIMessage不会走向tool调用，而是直接走向收敛回答的节点，这就导致没有生成tool_call.id对应的tool_call_id。

解决方案：在下一轮对话开始前，判断状态中自定义的循环或者工具调用次数是否超过自定义配置阈值，如果超过，就丢弃state，创建新的AIMessage给到llm，将请求达到上限的信息展示给用户

