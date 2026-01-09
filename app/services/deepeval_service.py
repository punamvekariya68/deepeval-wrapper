import os
import asyncio
import time
from typing import List, Dict, Any, Union, Optional
from datetime import datetime

# DeepEval imports
from deepeval import evaluate
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase, 
    MLLMTestCase,
    ArenaTestCase,
    Turn as DeepEvalTurn,
    ToolCall as DeepEvalToolCall,
    LLMTestCaseParams,
    MLLMImage as DeepEvalMLLMImage,
)
from deepeval.metrics import (
    # RAG Metrics
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    
    # Safety Metrics
    BiasMetric,
    ToxicityMetric,
    HallucinationMetric,
    PIILeakageMetric,
    
    # Task Metrics
    SummarizationMetric,
    ToolCorrectnessMetric,
    TaskCompletionMetric,
    JsonCorrectnessMetric,
    ArgumentCorrectnessMetric,
    
    # Behavioral Metrics
    RoleAdherenceMetric,
    RoleViolationMetric,
    NonAdviceMetric,
    MisuseMetric,
    PromptAlignmentMetric,
    KnowledgeRetentionMetric,
    
    # Conversational Metrics
    TurnRelevancyMetric,
    ConversationCompletenessMetric,
    
    # Custom Metrics
    GEval,
    ConversationalGEval,
    
    # Arena Metrics
    ArenaGEval,
    
    # Base classes
    BaseMetric,
    BaseConversationalMetric,
    BaseMultimodalMetric,
    BaseArenaMetric,
)
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.errors import MissingTestCaseParamsError

# Local imports
from ..models import (
    MetricType,
    MetricRequest,
    MetricResult,
    LLMTestCaseRequest,
    ConversationalTestCaseRequest,
    MLLMTestCaseRequest,
    ArenaTestCaseRequest,
    ToolCall,
    Turn,
    MLLMImage,
    TestCaseResult,
    EvaluationSummary,
    LLMTestCaseParam,
)
from ..config import settings


class DeepEvalService:
    """Service for interacting with DeepEval library."""
    
    def __init__(self):
        """Initialize the DeepEval service."""
        self._setup_environment()
        self._metric_registry = self._build_metric_registry()
    
    def _setup_environment(self):
        """Set up environment variables for LLM providers."""
        # if settings.openai_api_key:
        #     os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        # if settings.anthropic_api_key:
        #     os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        if settings.google_api_key:
            os.environ["GOOGLE_API_KEY"] = settings.google_api_key
        # if settings.cohere_api_key:
        #     os.environ["COHERE_API_KEY"] = settings.cohere_api_key
        if settings.deepeval_api_key:
            os.environ["DEEPEVAL_API_KEY"] = settings.deepeval_api_key
    
    def _build_metric_registry(self) -> Dict[MetricType, Dict[str, Any]]:
        """Build registry of available metrics with their metadata."""
        return {
            # RAG Metrics
            MetricType.FAITHFULNESS: {
                "class": FaithfulnessMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "rag"
            },
            MetricType.ANSWER_RELEVANCY: {
                "class": AnswerRelevancyMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "rag"
            },
            MetricType.CONTEXTUAL_PRECISION: {
                "class": ContextualPrecisionMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "rag"
            },
            MetricType.CONTEXTUAL_RECALL: {
                "class": ContextualRecallMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "rag"
            },
            MetricType.CONTEXTUAL_RELEVANCY: {
                "class": ContextualRelevancyMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "rag"
            },
            
            # Safety Metrics
            MetricType.BIAS: {
                "class": BiasMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "safety"
            },
            MetricType.TOXICITY: {
                "class": ToxicityMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "safety"
            },
            MetricType.HALLUCINATION: {
                "class": HallucinationMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "safety"
            },
            MetricType.PII_LEAKAGE: {
                "class": PIILeakageMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "safety"
            },
            
            # Task Metrics
            MetricType.SUMMARIZATION: {
                "class": SummarizationMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "task"
            },
            MetricType.TOOL_CORRECTNESS: {
                "class": ToolCorrectnessMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.TOOLS_CALLED, LLMTestCaseParams.EXPECTED_TOOLS],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "task"
            },
            MetricType.TASK_COMPLETION: {
                "class": TaskCompletionMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.TOOLS_CALLED],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "task"
            },
            MetricType.JSON_CORRECTNESS: {
                "class": JsonCorrectnessMetric,
                "required_params": [LLMTestCaseParams.ACTUAL_OUTPUT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "task"
            },
            MetricType.ARGUMENT_CORRECTNESS: {
                "class": ArgumentCorrectnessMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.TOOLS_CALLED],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "task"
            },
            
            # Behavioral Metrics
            MetricType.ROLE_ADHERENCE: {
                "class": RoleAdherenceMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "behavioral"
            },
            MetricType.ROLE_VIOLATION: {
                "class": RoleViolationMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "behavioral"
            },
            MetricType.NON_ADVICE: {
                "class": NonAdviceMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "behavioral"
            },
            MetricType.MISUSE: {
                "class": MisuseMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "behavioral"
            },
            MetricType.PROMPT_ALIGNMENT: {
                "class": PromptAlignmentMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "behavioral"
            },
            MetricType.KNOWLEDGE_RETENTION: {
                "class": KnowledgeRetentionMetric,
                "required_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "behavioral"
            },
            
            # Conversational Metrics
            MetricType.TURN_RELEVANCY: {
                "class": TurnRelevancyMetric,
                "required_params": [],  # Uses full conversation
                "supports_multimodal": False,
                "supports_conversational": True,
                "category": "conversational"
            },
            MetricType.CONVERSATION_COMPLETENESS: {
                "class": ConversationCompletenessMetric,
                "required_params": [],  # Uses full conversation
                "supports_multimodal": False,
                "supports_conversational": True,
                "category": "conversational"
            },
            
            # Custom Metrics
            MetricType.G_EVAL: {
                "class": GEval,
                "required_params": [],  # Configurable
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "custom"
            },
            MetricType.CONVERSATIONAL_G_EVAL: {
                "class": ConversationalGEval,
                "required_params": [],  # Configurable
                "supports_multimodal": False,
                "supports_conversational": True,
                "category": "custom"
            },
            
            # Arena Metrics
            MetricType.ARENA_G_EVAL: {
                "class": ArenaGEval,
                "required_params": [],  # Uses arena test case
                "supports_multimodal": False,
                "supports_conversational": False,
                "category": "arena"
            },
        }
    
    def create_metric(self, metric_request: MetricRequest) -> Union[BaseMetric, BaseConversationalMetric, BaseMultimodalMetric, BaseArenaMetric]:
        """Create a DeepEval metric instance from request."""
        metric_type = metric_request.metric_type
        
        if metric_type not in self._metric_registry:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        
        metric_info = self._metric_registry[metric_type]
        metric_class = metric_info["class"]
        
        # Common parameters
        common_params = {
            "threshold": metric_request.threshold or 0.5,
            "include_reason": metric_request.include_reason or True,
            "async_mode": metric_request.async_mode or True,
            "strict_mode": metric_request.strict_mode or False,
            "verbose_mode": metric_request.verbose_mode or False,
        }
        
        # Add model parameter if provided
        if metric_request.model:
            common_params["model"] = metric_request.model
        
        # Metric-specific parameters
        if metric_type == MetricType.FAITHFULNESS:
            if metric_request.truths_extraction_limit:
                common_params["truths_extraction_limit"] = metric_request.truths_extraction_limit
        
        elif metric_type in [MetricType.G_EVAL, MetricType.CONVERSATIONAL_G_EVAL, MetricType.ARENA_G_EVAL]:
            if not metric_request.name:
                raise ValueError(f"{metric_type} requires 'name' parameter")
            
            # Enforce mutual exclusivity between criteria and evaluation_steps
            if metric_request.criteria and metric_request.evaluation_steps:
                raise ValueError(f"{metric_type} can only use either 'criteria' OR 'evaluation_steps', not both")
            
            if not metric_request.criteria and not metric_request.evaluation_steps:
                raise ValueError(f"{metric_type} requires either 'criteria' OR 'evaluation_steps' parameter")
            
            # Handle different parameter types for different G-Eval variants
            evaluation_params = []
            if metric_request.evaluation_params:
                # Validate evaluation_params is not empty
                if len(metric_request.evaluation_params) == 0:
                    raise ValueError(f"{metric_type} evaluation_params cannot be an empty list")
                if metric_type == MetricType.CONVERSATIONAL_G_EVAL:
                    # Conversational G-Eval uses TurnParams - for now we'll map to common ones
                    # This is a simplified mapping - in a full implementation you'd import TurnParams
                    from deepeval.test_case import TurnParams
                    turn_param_map = {
                        LLMTestCaseParam.INPUT: TurnParams.CONTENT,  # Map input to content
                        LLMTestCaseParam.ACTUAL_OUTPUT: TurnParams.CONTENT,
                        LLMTestCaseParam.CONTEXT: TurnParams.RETRIEVAL_CONTEXT,
                        LLMTestCaseParam.RETRIEVAL_CONTEXT: TurnParams.RETRIEVAL_CONTEXT,
                        LLMTestCaseParam.TOOLS_CALLED: TurnParams.TOOLS_CALLED,
                    }
                    # Use TurnParams.CONTENT as default for conversational
                    evaluation_params = [turn_param_map.get(p, TurnParams.CONTENT) for p in metric_request.evaluation_params]
                else:
                    # Regular G-Eval uses LLMTestCaseParams
                    param_map = {
                        LLMTestCaseParam.INPUT: LLMTestCaseParams.INPUT,
                        LLMTestCaseParam.ACTUAL_OUTPUT: LLMTestCaseParams.ACTUAL_OUTPUT,
                        LLMTestCaseParam.EXPECTED_OUTPUT: LLMTestCaseParams.EXPECTED_OUTPUT,
                        LLMTestCaseParam.CONTEXT: LLMTestCaseParams.CONTEXT,
                        LLMTestCaseParam.RETRIEVAL_CONTEXT: LLMTestCaseParams.RETRIEVAL_CONTEXT,
                        LLMTestCaseParam.TOOLS_CALLED: LLMTestCaseParams.TOOLS_CALLED,
                        LLMTestCaseParam.EXPECTED_TOOLS: LLMTestCaseParams.EXPECTED_TOOLS,
                    }
                    evaluation_params = [param_map[p] for p in metric_request.evaluation_params if p in param_map]
            
            # G-Eval metrics use different parameters - filter out unsupported common params
            g_eval_params = {
                "name": metric_request.name,
            }
            
            # Set default evaluation_params based on metric type
            if metric_type == MetricType.CONVERSATIONAL_G_EVAL:
                from deepeval.test_case import TurnParams
                # ConversationalGEval defaults to [CONTENT, ROLE] and auto-adds them if missing
                default_params = [TurnParams.CONTENT, TurnParams.ROLE]
                g_eval_params["evaluation_params"] = evaluation_params or default_params
            else:
                g_eval_params["evaluation_params"] = evaluation_params or [LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
            
            # Add either criteria or evaluation_steps (mutually exclusive)
            if metric_request.criteria:
                # Validate criteria is not empty
                if not metric_request.criteria.strip():
                    raise ValueError(f"{metric_type} criteria cannot be an empty string")
                g_eval_params["criteria"] = metric_request.criteria
            elif metric_request.evaluation_steps:
                # Validate evaluation_steps is not empty
                if len(metric_request.evaluation_steps) == 0:
                    raise ValueError(f"{metric_type} evaluation_steps cannot be an empty list")
                g_eval_params["evaluation_steps"] = metric_request.evaluation_steps
            
            # Add rubric support if provided
            if metric_request.rubric:
                # Convert dict rubrics to proper Rubric objects (simplified version)
                g_eval_params["rubric"] = metric_request.rubric
            
            # Only add supported parameters for G-Eval
            if metric_request.threshold is not None:
                g_eval_params["threshold"] = metric_request.threshold
            if metric_request.model:
                g_eval_params["model"] = metric_request.model
            if metric_request.async_mode is not None:
                g_eval_params["async_mode"] = metric_request.async_mode
            if metric_request.strict_mode is not None:
                g_eval_params["strict_mode"] = metric_request.strict_mode
            if metric_request.verbose_mode is not None:
                g_eval_params["verbose_mode"] = metric_request.verbose_mode
            
            # Replace common_params with G-Eval specific params
            common_params = g_eval_params
        
        elif metric_type == MetricType.TOOL_CORRECTNESS:
            # Tool Correctness metric doesn't use LLMs, so filter out unsupported params
            tool_correctness_params = {
                "threshold": metric_request.threshold or 0.5,
                "include_reason": metric_request.include_reason or True,
                "strict_mode": metric_request.strict_mode or False,
                "verbose_mode": metric_request.verbose_mode or False,
            }
            
            # Add tool correctness specific parameters
            if metric_request.exact_match_tool_names is not None:
                tool_correctness_params["exact_match_tool_names"] = metric_request.exact_match_tool_names
            if metric_request.exact_match_input_parameters is not None:
                tool_correctness_params["exact_match_input_parameters"] = metric_request.exact_match_input_parameters
            if metric_request.exact_match_tool_output is not None:
                tool_correctness_params["exact_match_tool_output"] = metric_request.exact_match_tool_output
            
            # Replace common_params with tool correctness specific params
            common_params = tool_correctness_params
        
        elif metric_type == MetricType.SUMMARIZATION:
            if metric_request.assessment_questions:
                common_params["assessment_questions"] = metric_request.assessment_questions
        
        elif metric_type == MetricType.NON_ADVICE:
            if metric_request.advice_types:
                common_params["advice_types"] = metric_request.advice_types
            else:
                # Provide default advice types if none specified
                common_params["advice_types"] = ["financial", "medical", "legal"]
        
        elif metric_type == MetricType.MISUSE:
            if metric_request.domain:
                common_params["domain"] = metric_request.domain
            else:
                # Provide default domain if none specified
                common_params["domain"] = "general"
        
        elif metric_type == MetricType.BIAS:
            if metric_request.bias_types:
                common_params["bias_types"] = metric_request.bias_types
        
        elif metric_type == MetricType.TOXICITY:
            if metric_request.toxicity_categories:
                common_params["toxicity_categories"] = metric_request.toxicity_categories
        
        elif metric_type == MetricType.ROLE_VIOLATION:
            if metric_request.role:
                common_params["role"] = metric_request.role
            else:
                # Provide default role if none specified
                common_params["role"] = "helpful assistant"
        

        elif metric_type == MetricType.PROMPT_ALIGNMENT:
            if metric_request.prompt_instructions:
                common_params["prompt_instructions"] = metric_request.prompt_instructions
            else:
                # Provide default instructions if none specified
                common_params["prompt_instructions"] = "Follow the given instructions exactly"
        
        # Handle conversational metrics that may not support all common parameters
        elif metric_type in [MetricType.TURN_RELEVANCY, MetricType.CONVERSATION_COMPLETENESS, MetricType.KNOWLEDGE_RETENTION, MetricType.ROLE_ADHERENCE]:
            # These metrics may have specific parameter requirements, filter common params
            conversational_params = {
                "threshold": metric_request.threshold or 0.5,
                "include_reason": metric_request.include_reason or True,
                "strict_mode": metric_request.strict_mode or False,
                "verbose_mode": metric_request.verbose_mode or False
            }
            
            # Add model parameter if supported
            if metric_request.model:
                conversational_params["model"] = metric_request.model
            
            # Add async_mode if supported
            if metric_request.async_mode is not None:
                conversational_params["async_mode"] = metric_request.async_mode
            
            # Note: RoleAdherenceMetric doesn't accept role parameter - it infers role from conversation
                
            return metric_class(**conversational_params)
        
        # Add any additional custom parameters
        if metric_request.additional_params:
            common_params.update(metric_request.additional_params)
        
        try:
            return metric_class(**common_params)
        except Exception as e:
            raise ValueError(f"Failed to create metric {metric_type}: {str(e)}")
    
    def create_test_case(self, test_case_request) -> Union[LLMTestCase, ConversationalTestCase, MLLMTestCase, ArenaTestCase]:
        """Create a DeepEval test case from request."""
        if isinstance(test_case_request, LLMTestCaseRequest):
            return self._create_llm_test_case(test_case_request)
        elif isinstance(test_case_request, ConversationalTestCaseRequest):
            return self._create_conversational_test_case(test_case_request)
        elif isinstance(test_case_request, MLLMTestCaseRequest):
            return self._create_mllm_test_case(test_case_request)
        elif isinstance(test_case_request, ArenaTestCaseRequest):
            return self._create_arena_test_case(test_case_request)
        else:
            raise ValueError(f"Unsupported test case type: {type(test_case_request)}")
    
    def _create_llm_test_case(self, request: LLMTestCaseRequest) -> LLMTestCase:
        """Create LLM test case."""
        tools_called = None
        expected_tools = None
        
        if request.tools_called:
            tools_called = [self._convert_tool_call(tool) for tool in request.tools_called]
        
        if request.expected_tools:
            expected_tools = [self._convert_tool_call(tool) for tool in request.expected_tools]
        
        return LLMTestCase(
            input=request.input,
            actual_output=request.actual_output,
            expected_output=request.expected_output,
            context=request.context,
            retrieval_context=request.retrieval_context,
            tools_called=tools_called,
            expected_tools=expected_tools,
            name=request.name,
            additional_metadata=request.additional_metadata,
            comments=request.comments,
            tags=request.tags,
        )
    
    def _create_conversational_test_case(self, request: ConversationalTestCaseRequest) -> ConversationalTestCase:
        """Create conversational test case."""
        turns = []
        for turn_request in request.turns:
            tools_called = None
            if turn_request.tools_called:
                tools_called = [self._convert_tool_call(tool) for tool in turn_request.tools_called]
            
            # Build turn with only non-None values
            turn_params = {
                "role": turn_request.role,
                "content": turn_request.content
            }
            
            if turn_request.scenario:
                turn_params["scenario"] = turn_request.scenario
            if turn_request.expected_outcome:
                turn_params["expected_outcome"] = turn_request.expected_outcome
            if turn_request.retrieval_context:
                turn_params["retrieval_context"] = turn_request.retrieval_context
            if tools_called:
                turn_params["tools_called"] = tools_called
                
            turn = DeepEvalTurn(**turn_params)
            turns.append(turn)
        
        # Build ConversationalTestCase with only non-None values
        conversational_params = {
            "turns": turns
        }
        
        if request.chatbot_role:
            conversational_params["chatbot_role"] = request.chatbot_role
        if request.scenario:
            conversational_params["scenario"] = request.scenario
        if request.user_description:
            conversational_params["user_description"] = request.user_description
        if request.expected_outcome:
            conversational_params["expected_outcome"] = request.expected_outcome
        if request.context:
            conversational_params["context"] = request.context
        if request.name:
            conversational_params["name"] = request.name
        if request.additional_metadata:
            conversational_params["additional_metadata"] = request.additional_metadata
        
        if request.comments:
            conversational_params["comments"] = request.comments
        if request.tags:
            conversational_params["tags"] = request.tags
        
        return ConversationalTestCase(**conversational_params)
    
    def _create_mllm_test_case(self, request: MLLMTestCaseRequest) -> MLLMTestCase:
        """Create multimodal test case."""
        input_items = []
        for item in request.input:
            if isinstance(item, str):
                input_items.append(item)
            elif isinstance(item, MLLMImage):
                input_items.append(DeepEvalMLLMImage(url=item.url))
            else:
                input_items.append(str(item))
        
        tools_called = None
        expected_tools = None
        
        if request.tools_called:
            tools_called = [self._convert_tool_call(tool) for tool in request.tools_called]
        
        if request.expected_tools:
            expected_tools = [self._convert_tool_call(tool) for tool in request.expected_tools]
        
        return MLLMTestCase(
            input=input_items,
            actual_output=request.actual_output,
            expected_output=request.expected_output,
            context=request.context,
            retrieval_context=request.retrieval_context,
            tools_called=tools_called,
            expected_tools=expected_tools,
            name=request.name,
            additional_metadata=request.additional_metadata,
            comments=request.comments,
            tags=request.tags,
        )
    
    def _create_arena_test_case(self, request: ArenaTestCaseRequest) -> ArenaTestCase:
        """Create arena test case."""
        return ArenaTestCase(
            input=request.input,
            model_a_output=request.model_a_output,
            model_b_output=request.model_b_output,
            model_a_name=request.model_a_name,
            model_b_name=request.model_b_name,
            name=request.name,
            additional_metadata=request.additional_metadata,
            comments=request.comments,
            tags=request.tags,
        )
    
    def _convert_tool_call(self, tool: ToolCall) -> DeepEvalToolCall:
        """Convert API ToolCall to DeepEval ToolCall."""
        return DeepEvalToolCall(
            name=tool.name,
            description=tool.description,
            reasoning=tool.reasoning,
            output=tool.output,
            input_parameters=tool.input_parameters,
        )
    
    async def evaluate_single(
        self, 
        test_case_request,
        metric_requests: List[MetricRequest]
    ) -> TestCaseResult:
        """Evaluate a single test case with multiple metrics."""
        start_time = time.time()
        test_case = self.create_test_case(test_case_request)
        metric_results = []
        
        for metric_request in metric_requests:
            try:
                metric = self.create_metric(metric_request)
                result = await self._evaluate_metric_async(metric, test_case)
                metric_results.append(result)
            except Exception as e:
                error_result = MetricResult(
                    metric_type=metric_request.metric_type.value,
                    score=0.0,
                    threshold=metric_request.threshold or 0.5,
                    success=False,
                    error=str(e)
                )
                metric_results.append(error_result)
        
        overall_success = all(result.success for result in metric_results)
        execution_time = time.time() - start_time
        
        return TestCaseResult(
            test_case=test_case_request,
            metrics=metric_results,
            overall_success=overall_success,
            execution_time=execution_time,
        )
    
    async def evaluate_bulk(
        self,
        test_case_requests: List,
        metric_requests: List[MetricRequest],
        max_concurrent: int = 10
    ) -> Dict[str, Any]:
        """Evaluate multiple test cases with multiple metrics."""
        start_time = time.time()
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(test_case_request):
            async with semaphore:
                return await self.evaluate_single(test_case_request, metric_requests)
        
        # Execute evaluations concurrently
        tasks = [evaluate_with_semaphore(tc) for tc in test_case_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result for failed test case
                error_result = TestCaseResult(
                    test_case=test_case_requests[i],
                    metrics=[MetricResult(
                        metric_type="error",
                        score=0.0,
                        threshold=0.0,
                        success=False,
                        error=str(result)
                    )],
                    overall_success=False,
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)
        
        # Calculate summary
        total_execution_time = time.time() - start_time
        summary = self._calculate_summary(valid_results, total_execution_time)
        
        return {
            "results": valid_results,
            "summary": summary,
        }
    
    async def _evaluate_metric_async(self, metric, test_case) -> MetricResult:
        """Evaluate a single metric asynchronously."""
        try:
            # Use DeepEval's native async if available
            if hasattr(metric, 'a_measure'):
                await metric.a_measure(test_case)
            else:
                # Fallback to thread pool for sync-only metrics
                await asyncio.to_thread(metric.measure, test_case)
            
            return MetricResult(
                metric_type=metric.__class__.__name__.replace("Metric", "").lower(),
                score=metric.score,
                threshold=metric.threshold,
                success=metric.is_successful(),
                reason=getattr(metric, 'reason', None),
                score_breakdown=getattr(metric, 'score_breakdown', None),
                evaluation_model=getattr(metric, 'evaluation_model', None),
                evaluation_cost=getattr(metric, 'evaluation_cost', None),
                verbose_logs=getattr(metric, 'verbose_logs', None),
            )
        
        except MissingTestCaseParamsError as e:
            return MetricResult(
                metric_type=metric.__class__.__name__.replace("Metric", "").lower(),
                score=0.0,
                threshold=metric.threshold,
                success=False,
                error=f"Missing required parameters: {str(e)}"
            )
        except Exception as e:
            return MetricResult(
                metric_type=metric.__class__.__name__.replace("Metric", "").lower(),
                score=0.0,
                threshold=getattr(metric, 'threshold', 0.5),
                success=False,
                error=str(e)
            )
    
    def _calculate_summary(self, results: List[TestCaseResult], execution_time: float) -> EvaluationSummary:
        """Calculate summary statistics for evaluation results."""
        total_tests = len(results)
        successful_tests = sum(1 for result in results if result.overall_success)
        failed_tests = total_tests - successful_tests
        
        # Calculate per-metric summaries
        metric_summaries = {}
        for result in results:
            for metric_result in result.metrics:
                metric_type = metric_result.metric_type
                if metric_type not in metric_summaries:
                    metric_summaries[metric_type] = {
                        "scores": [],
                        "successes": 0,
                        "total": 0,
                        "errors": 0,
                    }
                
                summary = metric_summaries[metric_type]
                summary["total"] += 1
                
                if metric_result.error:
                    summary["errors"] += 1
                else:
                    summary["scores"].append(metric_result.score)
                    if metric_result.success:
                        summary["successes"] += 1
        
        # Calculate final metric statistics
        for metric_type, summary in metric_summaries.items():
            scores = summary["scores"]
            if scores:
                summary["average_score"] = sum(scores) / len(scores)
                summary["min_score"] = min(scores)
                summary["max_score"] = max(scores)
            else:
                summary["average_score"] = 0.0
                summary["min_score"] = 0.0
                summary["max_score"] = 0.0
            
            summary["success_rate"] = summary["successes"] / summary["total"] if summary["total"] > 0 else 0.0
            summary["error_rate"] = summary["errors"] / summary["total"] if summary["total"] > 0 else 0.0
            
            # Remove internal tracking fields
            del summary["scores"]
        
        return EvaluationSummary(
            total_test_cases=total_tests,
            successful_test_cases=successful_tests,
            failed_test_cases=failed_tests,
            success_rate=successful_tests / total_tests if total_tests > 0 else 0.0,
            total_execution_time=execution_time,
            metric_summaries=metric_summaries,
        )
    
    def get_metric_info(self, metric_type: MetricType) -> Dict[str, Any]:
        """Get information about a specific metric."""
        if metric_type not in self._metric_registry:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        info = self._metric_registry[metric_type].copy()
        info["metric_type"] = metric_type
        info["name"] = metric_type.value.replace("_", " ").title()
        
        return info
    
    def list_available_metrics(self) -> List[Dict[str, Any]]:
        """List all available metrics with their information."""
        return [self.get_metric_info(metric_type) for metric_type in self._metric_registry.keys()]
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of DeepEval service."""
        try:
            import deepeval
            deepeval_available = True
            deepeval_version = getattr(deepeval, '__version__', 'unknown')
        except ImportError:
            deepeval_available = False
            deepeval_version = None
        
        return {
            "deepeval_available": deepeval_available,
            "deepeval_version": deepeval_version,
            # "openai_configured": bool(settings.openai_api_key),
            # "anthropic_configured": bool(settings.anthropic_api_key),
            "google_configured": bool(settings.google_api_key),
            # "supported_metrics": len(self._metric_registry),
        }
