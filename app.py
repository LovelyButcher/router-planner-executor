# app.py - FIXED VERSION
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

# ---------- LLM Agent (FIXED) ----------
class LLM_Agent:
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        # Using a smaller, compatible model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def ask(self, question, max_length=200):
        inputs = self.tokenizer.encode(question + self.tokenizer.eos_token, 
                                       return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.95,
                temperature=0.7
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean the response
        answer = answer.replace(question, "").strip()
        # Simulate confidence for routing
        return answer, torch.randn(1, 1000)  # Random logits for demo

# ---------- Router ----------
class Router:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def confidence(self, logits):
        # Simplified confidence calculation
        return torch.sigmoid(logits.mean()).item()

    def route(self, confidence):
        return "llm" if confidence >= self.threshold else "lrm"

# ---------- Tools ----------
class PythonREPL:
    def __init__(self):
        self.name = "PythonREPL"
        self.description = "Executes Python code and returns the output"
    
    def run(self, code):
        try:
            exec_globals = {}
            exec(code, exec_globals)
            # Return only the last expression if any
            return str(exec_globals.get('result', 'Code executed successfully'))
        except Exception as e:
            return f"Error: {str(e)}"

class MathSolver:
    def __init__(self):
        self.name = "MathSolver"
        self.description = "Solves mathematical expressions"
    
    def run(self, expr):
        try:
            # IMPORTANT: Use safe eval or restrict to math operations only
            # This is a demo - in production, use a safer evaluator
            allowed_names = {
                'abs': abs, 'min': min, 'max': max, 'sum': sum,
                'len': len, 'round': round, 'int': int, 'float': float
            }
            # Remove dangerous operations
            expr = expr.replace("import", "")
            expr = expr.replace("open(", "")
            return str(eval(expr, {"__builtins__": {}}, allowed_names))
        except Exception as e:
            return f"Error: {str(e)}"

class WebSearch:
    def __init__(self):
        self.name = "WebSearch"
        self.description = "Searches the web for information"
    
    def run(self, query):
        return f"Simulated search result for: {query}\n1. Result 1: Information about {query}\n2. Result 2: More details on {query}"

class RAGKnowledge:
    def __init__(self, knowledge_list):
        self.name = "RAGKnowledge"
        self.description = "Queries knowledge base"
        self.knowledge = knowledge_list
    
    def run(self, question):
        # Simple similarity search simulation
        return f"According to knowledge base: {self.knowledge[0] if self.knowledge else 'No information found'}"

# ---------- Simple Planner-Executor ----------
class SimplePlanner:
    def create_plan(self, task):
        """Simple rule-based planner"""
        task_lower = task.lower()
        if "python" in task_lower or "code" in task_lower or "def " in task_lower:
            return {"tool": "python_repl", "action": task}
        elif "calculate" in task_lower or "math" in task_lower or "+" in task or "*" in task or "/" in task:
            # Extract math expression
            import re
            math_expr = re.findall(r'[\d\s\+\-\*\/\(\)\.]+', task)
            if math_expr:
                return {"tool": "math_solver", "action": math_expr[0]}
            return {"tool": "math_solver", "action": task}
        elif "search" in task_lower or "find" in task_lower or "what is" in task_lower:
            return {"tool": "web_search", "action": task}
        else:
            return {"tool": "rag_knowledge", "action": task}

class SimpleExecutor:
    def execute_plan(self, plan, tools):
        tool_name = plan["tool"]
        action = plan["action"]
        if tool_name in tools:
            result = tools[tool_name].run(action)
            return result, plan
        return f"Tool {tool_name} not found", plan

# ---------- LRM Agent (FIXED) ----------
class LRM_Agent:
    def __init__(self, tools):
        self.tools = tools
        self.planner = SimplePlanner()
        self.executor = SimpleExecutor()
    
    def plan_execute(self, task):
        plan = self.planner.create_plan(task)
        result, executed_plan = self.executor.execute_plan(plan, self.tools)
        return result, executed_plan

# ---------- Initialize ----------
def setup_app():
    # Create tools
    tools_dict = {
        "python_repl": PythonREPL(),
        "math_solver": MathSolver(),
        "web_search": WebSearch(),
        "rag_knowledge": RAGKnowledge(["Rome is in Italy", "Python is a programming language", "2+2=4"])
    }
    
    # Initialize agents
    llm = LLM_Agent()
    lrm = LRM_Agent(tools_dict)
    router = Router(threshold=0.5)  # Lower threshold for demo
    
    return llm, lrm, router, tools_dict

# ---------- Streamlit App ----------
def main():
    st.set_page_config(
        page_title="Router + Planner-Executor Chat Agent",
        layout="wide",
        page_icon="🤖"
    )
    
    st.title("🤖 Router + Planner-Executor Chat Agent")
    
    # Initialize with caching to prevent reloading on every interaction
    @st.cache_resource
    def load_agents():
        return setup_app()
    
    llm, lrm, router, tools = load_agents()
    
    # Session state
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_input" not in st.session_state:
        st.session_state.current_input = ""
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Chat input with form to prevent rerun issues
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Your message:", 
                key="chat_input",
                placeholder="Ask me anything...",
                value=st.session_state.current_input
            )
            submitted = st.form_submit_button("Send")
            
            if submitted and user_input:
                with st.spinner("Thinking..."):
                    # Get LLM response
                    llm_answer, logits = llm.ask(user_input)
                    confidence = router.confidence(logits)
                    route = router.route(confidence)
                    
                    # Route to appropriate agent
                    if route == "llm":
                        response = llm_answer
                        plan = "N/A (Direct LLM response)"
                    else:
                        response, plan = lrm.plan_execute(user_input)
                    
                    # Store in history
                    st.session_state.history.append({
                        "user": user_input,
                        "agent": response,
                        "route": route.upper(),
                        "confidence": f"{confidence:.2f}",
                        "plan": str(plan)
                    })
                    
                    # Clear input after processing
                    st.session_state.current_input = ""
                    st.rerun()
        
        # Display chat history
        st.markdown("---")
        st.subheader("Chat History")
        
        if not st.session_state.history:
            st.info("No messages yet. Start a conversation!")
        else:
            for i, msg in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Message {len(st.session_state.history)-i}: {msg['user'][:50]}...", expanded=i==0):
                    st.markdown(f"**You:** {msg['user']}")
                    st.markdown(f"**Agent [{msg['route']}]:** {msg['agent']}")
                    st.caption(f"Confidence: {msg['confidence']}")
                    if msg['plan'] != "N/A (Direct LLM response)":
                        st.caption(f"Plan: {msg['plan']}")
    
    with col2:
        st.subheader("🔧 Tools & Debug")
        
        if st.session_state.history:
            latest = st.session_state.history[-1]
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Route", latest['route'])
            with col_b:
                st.metric("Confidence", latest['confidence'])
            
            with st.expander("Plan Details"):
                st.text(latest['plan'])
        
        st.markdown("---")
        st.subheader("🛠️ Manual Tool Testing")
        
        # Tool testing interface
        tool_choice = st.selectbox("Select Tool:", list(tools.keys()))
        
        if tool_choice == "python_repl":
            code = st.text_area("Enter Python code:", "result = 2 + 2\nprint(result)", height=100)
            if st.button("Run Python"):
                result = tools[tool_choice].run(code)
                st.code(result)
        
        elif tool_choice == "math_solver":
            expr = st.text_input("Math expression:", "2 + 3 * 5")
            if st.button("Calculate"):
                result = tools[tool_choice].run(expr)
                st.success(f"Result: {result}")
        
        elif tool_choice == "web_search":
            query = st.text_input("Search query:", "artificial intelligence")
            if st.button("Search"):
                result = tools[tool_choice].run(query)
                st.info(result)
        
        elif tool_choice == "rag_knowledge":
            question = st.text_input("Question:", "Where is Rome?")
            if st.button("Query Knowledge"):
                result = tools[tool_choice].run(question)
                st.info(result)
        
        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.history = []
            st.rerun()

if __name__ == "__main__":
    main()
