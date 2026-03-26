from langchain_core.messages import HumanMessage

class ChatInterface:
    def __init__(self, rag_system):
        self.rag_system = rag_system

    def chat(self, message, history):
        if not self.rag_system.agent_graph:
            return "System not initialized!"
        
        try:
            cfg = self.rag_system.get_config()
            sanpshot = self.rag_system.agent_graph.get_state(cfg)
            if sanpshot.next:
                self.rag_system.agent_graph.update_state(cfg, {"messages": [HumanMessage(content=message.strip())]})
                result = self.rag_system.agent_graph.invoke(None, config=cfg)
            else:
                result = self.rag_system.agent_graph.invoke(
                    {
                        "messages": [
                            HumanMessage(content=message.strip())
                        ],
                    },
                    config=cfg,
                )
            return result["messages"][-1].content
        except Exception as e:
            return f"Error: {e}"
        finally:
            self.rag_system.observability.flush()

    def clear_session(self):
        self.rag_system.reset_thread()