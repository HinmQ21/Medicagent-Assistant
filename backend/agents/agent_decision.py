"""
Agent Decision System for Multi-Agent Medical Chatbot

This module handles the orchestration of different agents using LangGraph.
It dynamically routes user queries to the appropriate agent based on content and context.
"""

import json
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
import os, getpass
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.guardrails.local_guardrails import LocalGuardrails

from langgraph.checkpoint.memory import MemorySaver

import cv2
import numpy as np

from config import Config

load_dotenv()

# Load configuration
config = Config()

# Initialize memory
memory = MemorySaver()

# Specify a thread
thread_config = {"configurable": {"thread_id": "1"}}


# Agent that takes the decision of routing the request further to correct task specific agent
class AgentConfig:
    """Configuration settings for the agent decision system."""
    
    # Decision model
    DECISION_MODEL = "gemini-2.5-flash"  # or whichever model you prefer

    # Vision model for image analysis
    VISION_MODEL = "gemini-2.5-flash"
    
    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85
    
    # System instructions for the decision agent
    DECISION_SYSTEM_PROMPT = """You are an intelligent medical triage system that routes user queries to 
    the appropriate specialized agent. Your job is to analyze the user's request and determine which agent 
    is best suited to handle it based on the query content, presence of images, and conversation context.

    Available agents:
    1. CONVERSATION_AGENT - For general chat, greetings, and non-medical questions.
    2. RAG_AGENT - For specific medical knowledge questions that can be answered from established medical literature. Currently ingested medical knowledge involves 'introduction to brain tumor', 'deep learning techniques to diagnose and detect brain tumors', 'deep learning techniques to diagnose and detect covid / covid-19 from chest x-ray'.
    3. WEB_SEARCH_PROCESSOR_AGENT - For questions about recent medical developments, current outbreaks, or time-sensitive medical information.
    4. BRAIN_TUMOR_AGENT - For analysis of brain MRI images to detect and segment tumors.
    5. CHEST_XRAY_AGENT - For analysis of chest X-ray images to detect abnormalities.
    6. SKIN_LESION_AGENT - For analysis of skin lesion images to classify them as benign or malignant.
    7. BONE_FRACTURE_AGENT - For analysis of X-ray images to detect bone fractures and injuries.

    Make your decision based on these guidelines:
    - If the user has not uploaded any image, always route to the conversation agent.
    - If the user uploads a medical image, decide which medical vision agent is appropriate based on the image type and the user's query. If the image is uploaded without a query, always route to the correct medical vision agent based on the image type.
    - If the user asks about recent medical developments or current health situations, use the web search pocessor agent.
    - If the user asks specific medical knowledge questions, use the RAG agent.
    - For general conversation, greetings, or non-medical questions, use the conversation agent. But if image is uploaded, always go to the medical vision agents first.

    You must provide your answer in JSON format with the following structure:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
    """

    image_analyzer = ImageAnalysisAgent(config=config)


class AgentState(MessagesState):
    """State maintained across the workflow."""
    # messages: List[BaseMessage]  # Conversation history
    agent_name: Optional[str]  # Current active agent
    current_input: Optional[Union[str, Dict]]  # Input to be processed
    has_image: bool  # Whether the current input contains an image
    image_type: Optional[str]  # Type of medical image if present
    output: Optional[str]  # Final output to user
    needs_human_validation: bool  # Whether human validation is required
    retrieval_confidence: float  # Confidence in retrieval (for RAG agent)
    bypass_routing: bool  # Flag to bypass agent routing for guardrails
    insufficient_info: bool  # Flag indicating RAG response has insufficient information
    input_lang: str  # Detected language of the input


class AgentDecision(TypedDict):
    """Output structure for the decision agent."""
    agent: str
    reasoning: str
    confidence: float


def create_agent_graph():
    """Create and configure the LangGraph for agent orchestration."""

    # Initialize guardrails with the same LLM used elsewhere
    guardrails = LocalGuardrails(config.rag.llm)

    # LLM
    decision_model = config.agent_decision.llm
    
    # Initialize the output parser
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)
    
    # Create the decision prompt
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # Create the decision chain
    decision_chain = decision_prompt | decision_model | json_parser
    
    # Define graph state transformations
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        current_input = state["current_input"]
        has_image = False
        image_type = None
        
        # Get the text from the input
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # System now focuses on English only
        input_lang = 'en'  # Always English
        
        # Check input through guardrails if text is present
        if input_text:
            is_allowed, message = guardrails.check_input(input_text)
            if not is_allowed:
                # If input is blocked, return early with guardrail message
                print(f"Selected agent: INPUT GUARDRAILS, Message: ", message)
                
                # No translation needed - system is English-only
                
                return {
                    **state,
                    "messages": message,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True,  # flag to end flow
                    "input_lang": input_lang  # Store the detected language
                }
        
        # Original image processing code
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image", None)
            image_type_response = AgentConfig.image_analyzer.analyze_image(image_path)
            image_type = image_type_response['image_type']
            print("ANALYZED IMAGE TYPE: ", image_type)
        
        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "bypass_routing": False,  # Explicitly set to False for normal flow
            "input_lang": input_lang  # Store the detected language
        }
    
    def check_if_bypassing(state: AgentState) -> str:
        """Check if we should bypass normal routing due to guardrails."""
        if state.get("bypass_routing", False):
            return "apply_guardrails"
        return "route_to_agent"
    
    def route_to_agent(state: AgentState) -> Dict:
        """Make decision about which agent should handle the query."""
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history (last 3 messages)
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges (6 messages)  # Not provided control from config
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        decision_input = f"""
        User query: {input_text}

        Recent conversation context:
        {recent_context}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information, which agent should handle this query?
        """
        
        # Make the decision
        decision = decision_chain.invoke({"input": decision_input})

        # Decided agent
        print(f"Decision: {decision['agent']}")
        
        # Update state with decision
        updated_state = {
            **state,
            "agent_name": decision["agent"],
        }
        
        # Route based on agent name and confidence
        if decision["confidence"] < AgentConfig.CONFIDENCE_THRESHOLD:
            return {"agent_state": updated_state, "next": "needs_validation"}
        
        return {"agent_state": updated_state, "next": decision["agent"]}

    # Define agent execution functions (these will be implemented in their respective modules)
    def run_conversation_agent(state: AgentState) -> AgentState:
        """Handle general conversation."""

        print(f"Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]
        
        # System is English-only
        
        # Get query text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history
        recent_context = ""
        for msg in messages:#[-20:]:  # Get last 10 exchanges (20 messages)  # currently considering complete history - limit control from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        conversation_prompt = f"""User query: {input_text}

        Recent conversation context: {recent_context}

        You are an AI-powered Medical Conversation Assistant. Your goal is to facilitate smooth and informative conversations with users, handling both casual and medical-related queries. You must respond naturally while ensuring medical accuracy and clarity.

        ### Role & Capabilities
        - Engage in **general conversation** while maintaining professionalism.
        - Answer **medical questions** using verified knowledge.
        - Route **complex queries** to RAG (retrieval-augmented generation) or web search if needed.
        - Handle **follow-up questions** while keeping track of conversation context.
        - Redirect **medical images** to the appropriate AI analysis agent.

        ### Guidelines for Responding:
        1. **General Conversations:**
        - If the user engages in casual talk (e.g., greetings, small talk), respond in a friendly, engaging manner.
        - Keep responses **concise and engaging**, unless a detailed answer is needed.

        2. **Medical Questions:**
        - If you have **high confidence** in answering, provide a medically accurate response.
        - Ensure responses are **clear, concise, and factual**.

        3. **Follow-Up & Clarifications:**
        - Maintain conversation history for better responses.
        - If a query is unclear, ask **follow-up questions** before answering.

        4. **Handling Medical Image Analysis:**
        - Do **not** attempt to analyze images yourself.
        - If user speaks about analyzing or processing or detecting or segmenting or classifying any disease from any image, ask the user to upload the image so that in the next turn it is routed to the appropriate medical vision agents.
        - If an image was uploaded, it would have been routed to the medical computer vision agents. Read the history to know about the diagnosis results and continue conversation if user asks anything regarding the diagnosis.
        - After processing, **help the user interpret the results**.

        5. **Uncertainty & Ethical Considerations:**
        - If unsure, **never assume** medical facts.
        - Recommend consulting a **licensed healthcare professional** for serious medical concerns.
        - Avoid providing **medical diagnoses** or **prescriptions**—stick to general knowledge.

        ### Response Format:
        - Maintain a **conversational yet professional tone**.
        - Use **bullet points or numbered lists** for clarity when needed.
        - If pulling from external sources (RAG/Web Search), mention **where the information is from** (e.g., "According to Mayo Clinic...").
        - If a user asks for a diagnosis, remind them to **seek medical consultation**.

        ### Example User Queries & Responses:

        **User:** "Hey, how's your day going?"
        **You:** "I'm here and ready to help! How can I assist you today?"

        **User:** "I have a headache and fever. What should I do?"
        **You:** "I'm not a doctor, but headaches and fever can have various causes, from infections to dehydration. If your symptoms persist, you should see a medical professional."

        Conversational LLM Response:"""

        # print("Conversation Prompt:", conversation_prompt)

        response = config.conversation.llm.invoke(conversation_prompt)

        # No translation needed - system is English-only

        # print("Conversation respone:", response)

        # response = AIMessage(content="This would be handled by the conversation agent.")

        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }
    
    def run_rag_agent(state: AgentState) -> AgentState:
        """Handle medical knowledge queries using RAG."""
        # Initialize the RAG agent

        print(f"Selected agent: RAG_AGENT")

        rag_agent = MedicalRAG(config)
        
        messages = state["messages"]
        query = state["current_input"]
        rag_context_limit = config.rag.context_limit

        # System is English-only

        recent_context = ""
        for msg in messages[-rag_context_limit:]:# limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        response = rag_agent.process_query(query, chat_history=recent_context)
        retrieval_confidence = response.get("confidence", 0.0)  # Default to 0.0 if not provided

        print(f"Retrieval Confidence: {retrieval_confidence}")
        print(f"Sources: {len(response['sources'])}")

        # Check if response indicates insufficient information
        insufficient_info = False
        response_content = response["response"]
        
        # Extract the content properly based on type
        if isinstance(response_content, dict) and hasattr(response_content, 'content'):
            # If it's an AIMessage or similar object with a content attribute
            response_text = response_content.content
        else:
            # If it's already a string
            response_text = response_content
            
        print(f"Response text type: {type(response_text)}")
        print(f"Response text preview: {response_text[:100]}...")
        
        if isinstance(response_text, str) and (
            "I don't have enough information to answer this question based on the provided context" in response_text or 
            "I don't have enough information" in response_text or 
            "don't have enough information" in response_text.lower() or
            "not enough information" in response_text.lower() or
            "insufficient information" in response_text.lower() or
            "cannot answer" in response_text.lower() or
            "unable to answer" in response_text.lower()
            ):
            
            print("RAG response indicates insufficient information")
            print(f"Response text that triggered insufficient_info: {response_text[:100]}...")
            insufficient_info = True

        print(f"Insufficient info flag set to: {insufficient_info}")

        # No translation needed - system is English-only

        # Store RAG output ONLY if confidence is high
        if retrieval_confidence >= config.rag.min_retrieval_confidence:
            # response_output = response["response"]
            response_output = AIMessage(content=response_text)
        else:
            response_output = AIMessage(content="")
        
        return {
            **state,
            "output": response_output,
            "needs_human_validation": False,  # Assuming no validation needed for RAG responses
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT",
            "insufficient_info": insufficient_info
        }

    # Web Search Processor Node
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """Handles web search results, processes them with LLM, and generates a refined response."""

        print(f"Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        print("[WEB_SEARCH_PROCESSOR_AGENT] Processing Web Search Results...")
        
        messages = state["messages"]
        web_search_context_limit = config.web_search.context_limit

        # System is English-only

        recent_context = ""
        for msg in messages[-web_search_context_limit:]: # limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        web_search_processor = WebSearchProcessorAgent(config)

        processed_response = web_search_processor.process_web_search_results(query=state["current_input"], chat_history=recent_context)

        # No translation needed - system is English-only
        
        if state['agent_name'] != None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"

        # Overwrite any previous output with the processed Web Search response
        return {
            **state,
            # "output": "This would be handled by the web search agent, finding the latest information.",
            "output": processed_response,
            "agent_name": involved_agents
        }

    # Define Routing Logic
    def confidence_based_routing(state: AgentState) -> Dict[str, str]:
        """Route based on RAG confidence score and response content."""
        # Debug prints
        print(f"Routing check - Retrieval confidence: {state.get('retrieval_confidence', 0.0)}")
        print(f"Routing check - Insufficient info flag: {state.get('insufficient_info', False)}")
        
        # Redirect if confidence is low or if response indicates insufficient info
        if (state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence or 
            state.get("insufficient_info", False)):
            print("Re-routed to Web Search Agent due to low confidence or insufficient information...")
            return "WEB_SEARCH_PROCESSOR_AGENT"  # Correct format
        return "check_validation"  # No transition needed if confidence is high and info is sufficient
    
    def run_brain_tumor_agent(state: AgentState) -> AgentState:
        """Handle brain MRI image analysis."""

        print(f"Selected agent: BRAIN_TUMOR_AGENT")

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        # System is English-only

        if not image_path:
            response_text = "No image was provided for analysis. Please upload a brain MRI image."
            response = AIMessage(content=response_text)
            return {
                **state,
                "output": response,
                "needs_human_validation": False,
                "agent_name": "BRAIN_TUMOR_AGENT"
            }

        try:
            # Initialize the brain tumor analyzer
            from agents.image_analysis_agent.brain_tumor_agent.brain_tumor_inference import BrainTumorInference
            brain_tumor_analyzer = BrainTumorInference()

            # Analyze the MRI image
            analysis_results = brain_tumor_analyzer.analyze_mri(image_path)

            # Check if there was an error in analysis
            if 'error' in analysis_results:
                response_text = f"Error analyzing the image: {analysis_results['error']}"
                response = AIMessage(content=response_text)
                return {
                    **state,
                    "output": response,
                    "needs_human_validation": False,
                    "agent_name": "BRAIN_TUMOR_AGENT"
                }

            # Format the response based on analysis results
            confidence = analysis_results['confidence'] * 100
            
            if not analysis_results['has_tumor']:
                response_text = f"""Brain MRI Analysis Results:

🔍 **Tumor Detection**: NEGATIVE
📊 **Confidence**: {confidence:.1f}%
📝 **Recommendation**: {analysis_results['recommendation']}

ℹ️ **Note**: While no tumor was detected, regular medical check-ups are still recommended."""
            else:
                tumor_type = analysis_results['tumor_type']
                response_text = f"""Brain MRI Analysis Results:

🔍 **Tumor Detection**: POSITIVE
📋 **Tumor Type**: {tumor_type}
📊 **Confidence**: {confidence:.1f}%

📈 **Detailed Probabilities**:
{chr(10).join([f"- {class_name}: {prob*100:.1f}%" for class_name, prob in analysis_results['class_probabilities'].items()])}

📝 **Recommendation**: {analysis_results['recommendation']}

⚠️ **Important Note**: This is an AI-assisted analysis. Please consult with a medical professional for proper diagnosis and treatment."""

            # No translation needed - system is English-only

            response = AIMessage(content=response_text)

            return {
                **state,
                "output": response,
                "needs_human_validation": True,  # Medical diagnosis always needs validation
                "agent_name": "BRAIN_TUMOR_AGENT"
            }

        except Exception as e:
            error_text = f"An error occurred while analyzing the brain MRI image: {str(e)}"
            error_response = AIMessage(content=error_text)
            return {
                **state,
                "output": error_response,
                "needs_human_validation": False,
                "agent_name": "BRAIN_TUMOR_AGENT"
            }
    
    def run_chest_xray_agent(state: AgentState) -> AgentState:
        """Handle chest X-ray image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        print(f"Selected agent: CHEST_XRAY_AGENT")

        # System is English-only

        if not image_path:
            response_text = "No image was provided for analysis. Please upload a chest X-ray image."
            response = AIMessage(content=response_text)
            return {
                **state,
                "output": response,
                "needs_human_validation": False,
                "agent_name": "CHEST_XRAY_AGENT"
            }

        try:
            # classify chest x-ray into covid or normal
            predicted_class = AgentConfig.image_analyzer.classify_chest_xray(image_path)

            if predicted_class == "covid19":
                response_text = f"""Chest X-ray Analysis Results:

🔍 **COVID-19 Detection**: POSITIVE
📊 **Result**: The analysis indicates findings potentially consistent with COVID-19
📝 **Interpretation**: The chest X-ray shows patterns that may be associated with COVID-19 pneumonia

⚠️ **Important Medical Disclaimer**:
- This is an AI-assisted analysis and NOT a definitive medical diagnosis
- COVID-19 diagnosis requires clinical correlation with symptoms, exposure history, and laboratory tests (RT-PCR, antigen tests)
- Many conditions can cause similar X-ray findings
- Please consult with a qualified healthcare professional immediately for proper evaluation, additional testing, and appropriate medical management
- If you have symptoms or suspect COVID-19 exposure, follow local health guidelines for testing and isolation"""

            elif predicted_class == "normal":
                response_text = f"""Chest X-ray Analysis Results:

🔍 **COVID-19 Detection**: NEGATIVE
📊 **Result**: The analysis indicates NORMAL chest X-ray findings
📝 **Interpretation**: No obvious signs of COVID-19 pneumonia detected in this chest X-ray

ℹ️ **Important Notes**:
- A normal chest X-ray does not completely rule out COVID-19, especially in early stages or mild cases
- Many COVID-19 patients have normal chest X-rays, particularly in the early phase of infection
- Clinical symptoms and laboratory tests (RT-PCR, antigen tests) are more reliable for COVID-19 diagnosis

⚠️ **Medical Disclaimer**:
- This is an AI-assisted analysis and should not replace professional medical evaluation
- If you have COVID-19 symptoms or exposure concerns, please consult with a healthcare professional and consider appropriate testing
- Follow local health guidelines regardless of this X-ray analysis result"""

            else:
                response_text = f"""Chest X-ray Analysis Results:

❌ **Analysis Status**: INCONCLUSIVE
📝 **Issue**: The uploaded image is not clear enough for reliable analysis or may not be a valid chest X-ray image

🔧 **Recommendations**:
- Please ensure the image is a clear, high-quality chest X-ray
- The image should be properly oriented and well-lit
- Avoid blurry, cropped, or low-resolution images

⚠️ **Next Steps**: Please upload a clearer chest X-ray image or consult with a healthcare professional for proper medical evaluation."""

            # No translation needed - system is English-only

            response = AIMessage(content=response_text)

            return {
                **state,
                "output": response,
                "needs_human_validation": True,  # Medical diagnosis always needs validation
                "agent_name": "CHEST_XRAY_AGENT"
            }

        except Exception as e:
            error_text = f"An error occurred while analyzing the chest X-ray image: {str(e)}"
            error_response = AIMessage(content=error_text)
            return {
                **state,
                "output": error_response,
                "needs_human_validation": False,
                "agent_name": "CHEST_XRAY_AGENT"
            }
    
    def run_skin_lesion_agent(state: AgentState) -> AgentState:
        """Handle skin lesion image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        print(f"Selected agent: SKIN_LESION_AGENT")

        # System is English-only

        if not image_path:
            response_text = "No image was provided for analysis. Please upload a skin lesion image."
            response = AIMessage(content=response_text)
            return {
                **state,
                "output": response,
                "needs_human_validation": False,
                "agent_name": "SKIN_LESION_AGENT"
            }

        try:
            # Perform skin lesion segmentation
            predicted_mask = AgentConfig.image_analyzer.segment_skin_lesion(image_path)

            if predicted_mask:
                response_text = f"""Skin Lesion Analysis Results:

🔍 **Segmentation Status**: SUCCESSFUL
📊 **Analysis Type**: Automated lesion boundary detection and segmentation
📝 **Output**: Segmented visualization with lesion boundaries highlighted

🎯 **What This Analysis Shows**:
- **Lesion Boundaries**: The highlighted areas show the detected boundaries of the skin lesion
- **Segmentation Mask**: The overlay indicates the precise location and extent of the lesion
- **Spatial Analysis**: This helps assess lesion size, shape, and border characteristics

📋 **Clinical Relevance**:
- **Border Assessment**: Irregular or asymmetric borders may indicate concern
- **Size Measurement**: Accurate lesion dimensions for monitoring changes over time
- **Shape Analysis**: Helps evaluate lesion morphology and growth patterns
- **Documentation**: Provides baseline for future comparison and monitoring

⚠️ **Important Medical Disclaimer**:
- This is an AI-assisted **segmentation tool**, NOT a diagnostic system
- Segmentation does **NOT** determine if a lesion is benign or malignant
- This analysis **CANNOT** replace dermatological examination or biopsy
- Any concerning skin lesions require evaluation by a qualified dermatologist
- Changes in size, color, shape, or texture should be evaluated promptly by a healthcare professional

🏥 **Next Steps**:
- Consult a dermatologist for proper clinical evaluation
- Consider dermoscopy or biopsy if recommended by your healthcare provider
- Monitor any changes and seek immediate medical attention for rapid changes
- Use this segmentation as a reference for tracking lesion changes over time

📸 **Segmented Image**: The processed image with lesion boundaries is available for download and clinical reference."""

            else:
                response_text = f"""Skin Lesion Analysis Results:

❌ **Segmentation Status**: UNSUCCESSFUL
📝 **Issue**: Unable to detect or segment skin lesion in the uploaded image

🔧 **Possible Reasons**:
- Image quality may be insufficient (blurry, low resolution, poor lighting)
- The image may not contain a clear skin lesion
- Lesion may be too small or faint for automated detection
- Image orientation or cropping may affect analysis

💡 **Recommendations for Better Results**:
- Ensure good lighting and clear focus when taking the photo
- Capture the lesion with adequate surrounding normal skin for context
- Use high resolution and avoid excessive zoom or cropping
- Take the photo perpendicular to the skin surface
- Ensure the lesion is clearly visible and well-contrasted

⚠️ **Important Note**:
- Inability to segment does not mean the lesion is normal or abnormal
- Some lesions may be difficult for automated systems to detect
- **Always consult a dermatologist** for proper evaluation of any skin concerns
- Do not rely solely on automated analysis for medical decisions

🏥 **Next Steps**: Please consult with a dermatologist for professional evaluation, regardless of this automated analysis result."""

            # No translation needed - system is English-only

            response = AIMessage(content=response_text)

            return {
                **state,
                "output": response,
                "needs_human_validation": True,  # Medical diagnosis always needs validation
                "agent_name": "SKIN_LESION_AGENT"
            }

        except Exception as e:
            error_text = f"An error occurred while analyzing the skin lesion image: {str(e)}"
            error_response = AIMessage(content=error_text)
            return {
                **state,
                "output": error_response,
                "needs_human_validation": False,
                "agent_name": "SKIN_LESION_AGENT"
            }
    
    def run_bone_fracture_agent(state: AgentState) -> AgentState:
        """Handle bone fracture detection in X-ray images."""
        
        current_input = state["current_input"]
        image_path = current_input.get("image", None)
        
        print(f"Selected agent: BONE_FRACTURE_AGENT")
        
        # System is English-only
        
        if not image_path:
            response_text = "No image was provided for analysis. Please upload an X-ray image for bone fracture detection."
            response = AIMessage(content=response_text)
            return {
                **state,
                "output": response,
                "needs_human_validation": False,
                "agent_name": "BONE_FRACTURE_AGENT"
            }
        
        try:
            # Perform bone fracture detection
            detection_result = AgentConfig.image_analyzer.detect_bone_fracture(image_path)
            
            if detection_result['detections_found']:
                detection_count = detection_result['detection_count']
                avg_confidence = detection_result['average_confidence']
                
                response_text = f"""Bone & Medical Anomaly Detection Results:
                
🔍 **Detection Status**: MEDICAL ANOMALIES DETECTED
📊 **Analysis Type**: Automated medical abnormality detection using YOLOv8 deep learning model
📝 **Output**: X-ray image with detected anomalies highlighted in color-coded bounding boxes

🎯 **Detection Summary**:
- **Number of Anomalies Detected**: {detection_count}
- **Average Detection Confidence**: {avg_confidence:.2f} ({avg_confidence*100:.1f}%)
- **Model Type**: YOLOv8 trained on GRAZPEDWRI-DX pediatric wrist trauma dataset
- **Detection Threshold**: 25% confidence minimum
- **Detectable Classes**: Fractures, bone anomalies, lesions, foreign bodies, metal objects, soft tissue abnormalities

📋 **Clinical Analysis**:
- **Multi-Class Detection**: Color-coded bounding boxes indicate different types of detected anomalies
- **Confidence Scores**: Each detection includes a confidence percentage  
- **Spatial Mapping**: Precise coordinates of suspected medical anomalies
- **Classification**: Distinguishes between fractures, bone anomalies, lesions, foreign bodies, and other medical conditions
- **Documentation**: Annotated image available for medical reference

⚠️ **Important Medical Disclaimer**:
- This is an AI-assisted **detection tool**, NOT a diagnostic system
- AI detection does **NOT** replace clinical judgment or radiological interpretation
- False positives and false negatives are possible with any automated system
- This analysis **CANNOT** replace proper medical evaluation by qualified healthcare professionals
- Any suspected fractures require immediate evaluation by a radiologist or orthopedic specialist

🏥 **Recommended Next Steps**:
- Consult an orthopedic specialist or radiologist for professional interpretation
- Consider additional imaging (CT, MRI) if clinically indicated
- Seek immediate medical attention for severe trauma or suspected displaced fractures
- Use this AI analysis as a supplementary tool alongside clinical assessment
- Document findings and compare with clinical examination

📸 **Annotated Image**: The processed X-ray with detected fracture locations is available for download and clinical reference."""
            else:
                response_text = f"""Bone & Medical Anomaly Detection Results:

🔍 **Detection Status**: NO MEDICAL ANOMALIES DETECTED
📊 **Analysis Type**: Automated medical abnormality detection using YOLOv8 deep learning model
📝 **Output**: X-ray image analyzed with no significant abnormalities identified

🎯 **Analysis Summary**:
- **Medical Anomalies Detected**: None
- **Model Type**: YOLOv8 trained on GRAZPEDWRI-DX pediatric wrist trauma dataset
- **Detection Threshold**: 25% confidence minimum
- **Image Quality**: Successfully processed and analyzed
- **Scanned For**: Fractures, bone anomalies, lesions, foreign bodies, metal objects, soft tissue abnormalities

📋 **Clinical Interpretation**:
- **No Obvious Abnormalities**: AI model did not detect significant medical anomalies above threshold
- **Image Analysis**: Complete X-ray examination performed
- **Documentation**: Processed image available for medical reference

⚠️ **Important Medical Disclaimer**:
- **Absence of AI detection does NOT rule out fractures**
- Subtle, hairline, or stress fractures may not be detected by AI
- Clinical symptoms and examination findings remain paramount
- Some fracture types may require specialized imaging or expert interpretation
- This analysis **CANNOT** replace professional radiological evaluation

🏥 **Recommended Next Steps**:
- Clinical correlation with patient symptoms and physical examination
- Consider radiologist review if clinical suspicion remains high
- Follow-up imaging may be needed if symptoms persist
- Seek medical attention for continued pain, swelling, or functional impairment
- Use this AI analysis as a supplementary screening tool only

📸 **Processed Image**: The analyzed X-ray image is available for clinical reference."""
            
            response = AIMessage(content=response_text)
            return {
                **state,
                "output": response,
                "needs_human_validation": True,  # Medical diagnosis always needs validation
                "agent_name": "BONE_FRACTURE_AGENT"
            }
            
        except Exception as e:
            logger.error(f"Error in bone fracture detection: {e}")
            error_response = AIMessage(content=f"Error during bone fracture analysis: {str(e)}. Please try again or consult a medical professional.")
            return {
                **state,
                "output": error_response,
                "needs_human_validation": False,
                "agent_name": "BONE_FRACTURE_AGENT"
            }
    
    def handle_human_validation(state: AgentState) -> Dict:
        """Prepare for human validation if needed."""
        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}
    
    def perform_human_validation(state: AgentState) -> AgentState:
        """Handle human validation process."""
        print(f"Selected agent: HUMAN_VALIDATION")
        
        # System is English-only
        
        # Get the original output content
        output_content = state['output'].content
        
        # Create the validation prompt
        validation_prompt = f"{output_content}\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."
        
        # No translation needed - system is English-only
        
        # Create an AI message with the validation prompt
        validation_message = AIMessage(content=validation_prompt)

        return {
            **state,
            "output": validation_message,
            "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
        }

    # Check output through guardrails
    def apply_output_guardrails(state: AgentState) -> AgentState:
        """Apply output guardrails to the generated response."""
        output = state["output"]
        current_input = state["current_input"]

        # Check if output is valid
        if not output or not isinstance(output, (str, AIMessage)):
            return state

        output_text = output if isinstance(output, str) else output.content
        
        # System is English-only
        
        # If the last message was a human validation message
        if "Human Validation Required" in output_text:
            # Check if the current input is a human validation response
            validation_input = ""
            if isinstance(current_input, str):
                validation_input = current_input
            elif isinstance(current_input, dict):
                validation_input = current_input.get("text", "")
            
            # If validation input exists
            if validation_input.lower().startswith(('yes', 'no')):
                # Add the validation result to the conversation history
                validation_response = HumanMessage(content=f"Validation Result: {validation_input}")
                
                # If validation is 'No', modify the output
                if validation_input.lower().startswith('no'):
                    fallback_message_text = "The previous medical analysis requires further review. A healthcare professional has flagged potential inaccuracies."
                    
                    # No translation needed - system is English-only
                    
                    fallback_message = AIMessage(content=fallback_message_text)
                    return {
                        **state,
                        "messages": [validation_response, fallback_message],
                        "output": fallback_message
                    }
                
                return {
                    **state,
                    "messages": validation_response
                }
        
        # Get the original input text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Apply output sanitization
        sanitized_output = guardrails.check_output(output_text, input_text)
        
        # No translation needed - system is English-only
        
        # For non-validation cases, add the sanitized output to messages
        sanitized_message = AIMessage(content=sanitized_output) if isinstance(output, AIMessage) else sanitized_output
        
        return {
            **state,
            "messages": sanitized_message,
            "output": sanitized_message
        }

    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("BRAIN_TUMOR_AGENT", run_brain_tumor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("BONE_FRACTURE_AGENT", run_bone_fracture_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)
    
    # Define the edges (workflow connections)
    workflow.set_entry_point("analyze_input")
    # workflow.add_edge("analyze_input", "route_to_agent")
    # Add conditional routing for guardrails bypass
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "route_to_agent"
        }
    )
    
    # Connect decision router to agents
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "BRAIN_TUMOR_AGENT": "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT": "SKIN_LESION_AGENT",
            "BONE_FRACTURE_AGENT": "BONE_FRACTURE_AGENT",
            "needs_validation": "RAG_AGENT"  # Default to RAG if confidence is low
        }
    )
    
    # Connect agent outputs to validation check
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    # workflow.add_edge("RAG_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)
    workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")
    workflow.add_edge("BONE_FRACTURE_AGENT", "check_validation")

    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)
    
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "apply_guardrails"  # Route to guardrails instead of END
        }
    )
    
    # workflow.add_edge("human_validation", END)
    
    # Compile the graph
    return workflow.compile(checkpointer=memory)


def init_agent_state() -> AgentState:
    """Initialize the agent state with default values."""
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False,
        "input_lang": "en"
    }


# Language detection and translation functions removed - system now focuses on English only

def process_query(query: Union[str, Dict]) -> str:
    """
    Process a user query through the agent decision system.

    Args:
        query: User input (text string or dict with text and image)

    Returns:
        Response from the appropriate agent (English only)
    """
    # Initialize the graph
    graph = create_agent_graph()
    
    # Initialize state
    state = init_agent_state()
    
    # Extract query text and detect language
    if isinstance(query, dict):
        query_text = query.get("text", "")
    else:
        query_text = query

    # System now focuses on English only - no language detection or translation needed
    input_lang = 'en'  # Always English
    
    # Store the original language in the state
    state["input_lang"] = input_lang
    
    # Add the current query (now in English)
    state["current_input"] = query

    # Rewrite the query to be more clear and understandable
    if isinstance(query, dict):
        query_text = query.get("text", "")
    else:
        query_text = query

    # Only rewrite if there's actual text to rewrite
    if query_text:
        rewrite_prompt = f"""Please rewrite the following query in clear, simple English while maintaining its original meaning and intent:

        Original query: {query_text}

        Rewritten query:"""

        rewritten_query = config.conversation.llm.invoke(rewrite_prompt)
        
        # Update the query with the rewritten version
        if isinstance(query, dict):
            query["text"] = rewritten_query.content
        else:
            query = rewritten_query.content
            
        # Update state with rewritten query
        state["current_input"] = query

    # To handle image upload case
    if isinstance(query, dict):
        query = query.get("text", "") + ", user uploaded an image for diagnosis."
    
    state["messages"] = [HumanMessage(content=query)]

    # Process the query
    result = graph.invoke(state, thread_config)

    # Keep history to reasonable size
    if len(result["messages"]) > config.max_conversation_history:
        result["messages"] = result["messages"][-config.max_conversation_history:]

    # No translation needed - system is English-only

    # visualize conversation history in console
    for m in result["messages"]:
        m.pretty_print()
    
    # Add the response to conversation history
    return result