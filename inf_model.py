import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import asyncio # GraphRAG API calls might be async
import pandas as pd # GraphRAG API uses pandas DataFrames
from pathlib import Path # Needed for GraphRAG root_dir

# --- GraphRAG 引入 ---
# 尝试从 graphrag.api 导入核心查询函数 - 只保留 local_search
try:
    # Based on graphrag/api/query.py
    from graphrag.api import local_search # <<< Only import local_search
    # Need functions to load config and dataframes
    from graphrag.config.load_config import load_config
    from graphrag.utils.api import create_storage_from_config
    from graphrag.utils.storage import load_table_from_storage, storage_has_table
    # Also need the config model to hint the type
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    # # No longer importing BaseLogger here

    GRAPH_RAG_AVAILABLE = True
    # Use standard Python logger instance defined later
    # logger = logging.getLogger(__name__) # Will be re-get after basicConfig
    # logger.info("GraphRAG query API functions and utilities imported successfully.")

    # Helper function for loading dataframes (reused from backend logic)
    # This version is slightly simplified as it only needs required outputs for local search
    # Accepts Path object for root_dir
    async def load_graphrag_local_dataframes(root_dir: Path, config: GraphRagConfig) -> dict[str, pd.DataFrame | None]:
        """Loads required dataframes for Local Search."""
        dataframe_dict = {}
        # Hardcode required/optional outputs for Local Search as Global is removed
        required_outputs = ["communities", "community_reports", "text_units", "relationships", "entities"]
        optional_outputs = ["covariates"]

        # Assume single-index for simplicity unless config indicates multi-index
        if config.outputs and len(config.outputs) > 1:
            logger.warning("GraphRAG multi-index configuration detected. This backend implementation currently only supports single-index queries. Multi-index queries are not supported.")
            output_config = list(config.outputs.values())[0] # Use the first index as fallback
        elif config.output:
            output_config = config.output
        else:
            raise ValueError("GraphRAG configuration missing 'output' or 'outputs' section.")

        storage_obj = create_storage_from_config(output_config)

        # Load required outputs
        for name in required_outputs:
            logger.info(f"Loading required GraphRAG table for Local Search: {name}")
            try:
                df_value = await load_table_from_storage(name=name, storage=storage_obj)
                # Check if loaded dataframe is not None/empty
                if df_value is None or df_value.empty:
                     logger.warning(f"Loaded required table '{name}' is empty or None.")
                     if df_value is None: # If load_table_from_storage returned None
                         raise FileNotFoundError(f"Required Local Search table '{name}' could not be loaded (returned None).")

                dataframe_dict[name] = df_value
                logger.info(f"Successfully loaded required table: {name} ({len(df_value)} rows)")
            except FileNotFoundError:
                 logger.error(f"Required Local Search table '{name}' NOT FOUND in storage.", exc_info=True)
                 raise # Re-raise FileNotFoundError as it's critical for Local Search
            except Exception as e: # Catch ANY other exception during loading
                 logger.error(f"Error loading required Local Search table '{name}': {e}", exc_info=True)
                 raise Exception(f"Error loading required Local Search table '{name}': {str(e)}") from e # Re-raise other exceptions


        # Load optional outputs
        if optional_outputs:
            for optional_file in optional_outputs:
                logger.info(f"Checking for optional GraphRAG table: {optional_file}")
                try:
                    file_exists = await storage_has_table(optional_file, storage_obj)
                    if file_exists:
                        df_value = await load_table_from_storage(name=optional_file, storage=storage_obj)
                        dataframe_dict[optional_file] = df_value
                        logger.info(f"Loaded optional table: {optional_file} ({len(df_value) if df_value is not None else 'None'} rows)")
                    else:
                        dataframe_dict[optional_file] = None
                        logger.info(f"Optional table '{optional_file}' not found.")
                except Exception as e:
                     logger.warning(f"Failed to load optional GraphRAG table '{optional_file}': {e}", exc_info=True)
                     dataframe_dict[optional_file] = None # Set to None if loading fails


        # --- ADDED CHECK FOR REQUIRED DATAFRAMES ---
        # Verify all required dataframes were successfully loaded (not None and in dict)
        missing_dfs = [df_name for df_name in required_outputs if df_name not in dataframe_dict or dataframe_dict.get(df_name) is None]

        if missing_dfs:
             error_msg = f"缺少 GraphRAG 必要的 Local Search 数据文件或加载失败: {', '.join(missing_dfs)}。请确认 '{root_dir}' 目录已包含所有必要的 GraphRAG 索引构建后的输出文件。"
             logger.error(error_msg)
             # Raise a specific error that will be caught by the outer try/except
             raise FileNotFoundError(error_msg) # Use FileNotFoundError as it's closest to the issue
        # --- END ADDED CHECK ---


        return dataframe_dict


except ImportError as e:
    # This block will be executed if any of the required graphrag modules fail to import
    logger = logging.getLogger(__name__) # Ensure logger is available
    logger.error(f"无法导入 GraphRAG 核心组件: {e}", exc_info=True)
    logger.warning("GraphRAG 功能将不可用。请确保 graphrag 已正确安装，并且您安装的版本提供了 'graphrag.api' 中的 'local_search' 函数，以及必要的工具函数。")
    GRAPH_RAG_AVAILABLE = False
except Exception as e:
     # Catch any other exceptions during the initial import block (e.g., issues with pandas, asyncio)
     logger = logging.getLogger(__name__) # Ensure logger is available
     logger.error(f"导入 GraphRAG 或其依赖时发生未知错误: {e}", exc_info=True)
     GRAPH_RAG_AVAILABLE = False


# --- 配置 ---
# Qwen 模型配置
BASE_MODEL_DEFINITION_PATH = ".cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct"
ADAPTER_DEFINITION_PATH = "qwen2.5-1.5b-finetuned"
BASE_MODEL_CALCULATION_PATH = ".cache/modelscope/hub/models/Qwen/Qwen2___5-Math-1___5B-Instruct"
ADAPTER_CALCULATION_PATH = "qwen2.5-1.5b-math-finetuned-info-theory-loss-eval"

# GraphRAG 配置
# **请将这里的路径改为你的 GraphRAG 项目的实际根目录！**
GRAPH_RAG_ROOT_DIR = "inf_model_GraphRag/ragtest"
# 确保这个路径是相对于你运行这个Python脚本的目录，或者是一个绝对路径。


# --- Global variables ---
tokenizer_definition = None
model_definition = None
tokenizer_calculation = None
model_calculation = None
# GRAPH_RAG_AVAILABLE is declared globally in the try...except block above


# --- Flask App setup ---
app = Flask(__name__)
CORS(app) # Allow cross-origin requests from any source

# Re-configure logging after the initial imports if needed, ensure levels are set
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # Re-get logger after basicConfig if necessary

# # No longer creating a specific graphrag_logger instance here - using standard logger

def determine_device_and_dtype():
    """Determine the appropriate device and torch_dtype."""
    if torch.cuda.is_available():
        # Use index 0 for simplicity, or iterate/use device_map="auto"
        device_map_param = "auto" # Let HF handle distribution
        # Try bfloat16, otherwise use float16
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"Using CUDA, device_map='{device_map_param}', dtype={dtype}")
        # For inputs.to(), need a target device, usually cuda:0 or the first device
        target_device = "cuda" # Simplified assumption
    elif torch.backends.mps.is_available(): # For Apple Silicon
        device_map_param = "mps"
        dtype = torch.float16
        logger.info(f"Using MPS, device_map='{device_map_param}', dtype={dtype}")
        target_device = "mps"
    else:
        device_map_param = "cpu"
        dtype = None # CPU usually uses float32 (None)
        logger.info(f"Using CPU, device_map='{device_map_param}', dtype={dtype}")
        target_device = "cpu"
    return device_map_param, dtype, target_device


def load_individual_model_and_tokenizer(base_model_path: str, adapter_path: str, model_name_log: str):
    """
    Load a single base model, its tokenizer, and apply a PEFT adapter.
    Returns (tokenizer, model) on success, (tokenizer, None) or (None, None) on failure.
    """
    tokenizer_instance = None
    base_model_instance = None

    logger.info(f"Starting to load {model_name_log} Tokenizer from: {base_model_path}")
    try:
        tokenizer_instance = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        logger.info(f"{model_name_log} Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load {model_name_log} Tokenizer: {e}", exc_info=True)
        return None, None # Tokenizer failed, model cannot be used

    logger.info(f"Starting to load {model_name_log} base model from: {base_model_path}")
    device_map_param, torch_dtype, _ = determine_device_and_dtype() # Get device_map only
    try:
        base_model_instance = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map_param, # Use auto or specified device
            trust_remote_code=True
        )
        logger.info(f"{model_name_log} base model loaded successfully, device_map: {device_map_param}.")
        # Attempt to get device information - might be complex with device_map="auto"
        if hasattr(base_model_instance, 'device'):
             logger.info(f"{model_name_log} model deployed on device: {base_model_instance.device}")
        # For device_map="auto", the device might be a dictionary or a different object
        elif hasattr(base_model_instance, 'hf_device_map'):
             logger.info(f"{model_name_log} model device map: {base_model_instance.hf_device_map}")
        else:
             logger.info(f"{model_name_log} model device info not available (device_map='auto'?)")


    except Exception as e:
        logger.error(f"Failed to load {model_name_log} base model: {e}", exc_info=True)
        return tokenizer_instance, None # Return loaded tokenizer, but model load failed

    logger.info(f"Starting to load {model_name_log} PEFT adapter from: {adapter_path}")
    try:
        peft_model_instance = PeftModel.from_pretrained(base_model_instance, adapter_path)
        logger.info(f"{model_name_log} PEFT adapter loaded successfully.")
        if hasattr(peft_model_instance, 'device'):
             logger.info(f"{model_name_log} PEFT model deployed on device: {peft_model_instance.device}")
        elif hasattr(peft_model_instance, 'hf_device_map'):
             logger.info(f"{model_name_log} PEFT model device map: {peft_model_instance.hf_device_map}")
        else:
            logger.info(f"{model_name_log} PEFT model device info not available.")
        return tokenizer_instance, peft_model_instance
    except Exception as e:
        logger.error(f"Failed to load {model_name_log} PEFT adapter: {e}", exc_info=True)
        # If adapter fails, the task-specific model isn't ready.
        return tokenizer_instance, None # Return loaded tokenizer, but adapter failed


def load_all_models_and_tokenizers():
    """
    Load all specified models and their tokenizers.
    """
    global tokenizer_definition, model_definition, tokenizer_calculation, model_calculation
    global GRAPH_RAG_AVAILABLE # Need global declaration here as we might modify it

    logger.info("--- Starting to load Qwen Definition Model ---") # Updated Title
    tokenizer_definition, model_definition = load_individual_model_and_tokenizer(
        BASE_MODEL_DEFINITION_PATH, ADAPTER_DEFINITION_PATH, "Qwen Definition Model" # Updated Name
    )
    if not model_definition:
        logger.warning("Qwen Definition model failed to load fully. Requests depending on this model might fail.") # Updated Warning

    logger.info("--- Starting to load Qwen Calculation Model ---") # Updated Title
    tokenizer_calculation, model_calculation = load_individual_model_and_tokenizer(
        BASE_MODEL_CALCULATION_PATH, ADAPTER_CALCULATION_PATH, "Qwen Calculation Model" # Updated Name
    )
    if not model_calculation:
         logger.warning("Qwen Calculation model failed to load fully. Requests depending on this model might fail.") # Updated Warning

    # Check GraphRAG root directory existence, only if GraphRAG core components were imported
    if 'GRAPH_RAG_AVAILABLE' in globals() and GRAPH_RAG_AVAILABLE:
        logger.info(f"Checking GraphRAG project root directory: {GRAPH_RAG_ROOT_DIR}")
        # This check uses os.path.isdir which accepts string paths
        if not os.path.isdir(GRAPH_RAG_ROOT_DIR):
            logger.error(f"GraphRAG 项目根目录 '{GRAPH_RAG_ROOT_DIR}' 不存在或不是一个目录。")
            GRAPH_RAG_AVAILABLE = False # Disable GraphRAG if directory is missing
            logger.warning("GraphRAG functionality disabled due to missing root directory.")
        else:
             logger.info(f"GraphRAG project root directory found.")
    # If GRAPH_RAG_AVAILABLE was never set due to import errors, the above block is skipped.
    # Ensure it's set to False if import failed
    elif 'GRAPH_RAG_AVAILABLE' not in globals():
         GRAPH_RAG_AVAILABLE = False # Should be handled by the initial try/except, but defensive


    logger.info("All models and components load attempts completed.")


def get_loading_status_details():
    """
    Helper function to calculate the loading status of models and GraphRAG components.
    Returns a dictionary with status information.
    This function does NOT use Flask context and can be called outside request handling.
    """
    status_details = {}
    overall_status = "ok" # Initial assumption everything is okay

    # Need access to global variables
    global tokenizer_definition, model_definition, tokenizer_calculation, model_calculation, GRAPH_RAG_AVAILABLE, GRAPH_RAG_ROOT_DIR


    # Check Definition Model
    if tokenizer_definition and model_definition:
        status_details["definition_model"] = {"status": "loaded", "device": str(getattr(model_definition, 'device', 'N/A'))}
    elif tokenizer_instance and not model_definition: # Check tokenizer_instance for partial load
        status_details["definition_model"] = {"status": "tokenizer_loaded_model_or_adapter_failed"}
        overall_status = "degraded"
    else:
        status_details["definition_model"] = {"status": "not_loaded"}
        overall_status = "error" # A key component is missing

    # Check Calculation Model
    if tokenizer_calculation and model_calculation:
        status_details["calculation_model"] = {"status": "loaded", "device": str(getattr(model_calculation, 'device', 'N/A'))}
    elif tokenizer_calculation and not model_calculation: # Check tokenizer_calculation for partial load
        status_details["calculation_model"] = {"status": "tokenizer_loaded_model_or_adapter_failed"}
        if overall_status != "error": # Don't degrade from error state
            overall_status = "degraded"
        # else: overall_status remains error if it was error
    else:
        status_details["calculation_model"] = {"status": "not_loaded"}
        # overall_status is error if this key model is missing, but check if definition model was already error
        if overall_status != "error":
             overall_status = "error" # Set to error if not already error from definition model


    # Check GraphRAG (only Local Search capability is relevant now)
    graphrag_details = {
        "status": "unknown", # Explicit initial status
        "message": "GraphRAG availability not determined before checks.",
        "root_dir": GRAPH_RAG_ROOT_DIR, # Include path regardless of status
        "root_dir_status": "unchecked" # Explicit initial status
    }


    if 'GRAPH_RAG_AVAILABLE' not in globals():
         # Import failed or variable never set (less likely now with try/except)
         graphrag_details["status"] = "import_failed" # Assuming import failed if not in globals
         graphrag_details["message"] = "GraphRAG core components could not be imported (variable not in globals)."
         if overall_status != "error": # Import failed makes overall status degraded
              overall_status = "degraded"


    elif not GRAPH_RAG_AVAILABLE:
        # Variable exists but is False (import failed)
        graphrag_details["status"] = "import_failed"
        graphrag_details["message"] = "GraphRAG core components could not be imported."
        if overall_status != "error":
             overall_status = "degraded"

    elif GRAPH_RAG_AVAILABLE:
        # Components imported, check directory
        graphrag_details["status"] = "components_imported" # Intermediate status
        graphrag_details["message"] = "GraphRAG core components imported."
        graphrag_details["root_dir_status"] = "checking"
        if os.path.isdir(GRAPH_RAG_ROOT_DIR):
             graphrag_details["root_dir_status"] = "exists"
             graphrag_details["status"] = "local_search_potentially_available" # Final success status for local
             graphrag_details["message"] = "GraphRAG Local Search potentially available."
        else:
             # graphrag_root_dir_status = "missing" # Typo, unused variable
             graphrag_details["root_dir_status"] = "missing"
             graphrag_details["status"] = "config_error" # If imported but dir missing, it's config error
             graphrag_details["message"] = f"GraphRAG root directory '{GRAPH_RAG_ROOT_DIR}' not found."
             # if overall_status != "error": # Config error makes overall status error
             #     overall_status = "error" # This logic is handled below after graphrag_details is assigned
             # Re-evaluate overall_status after processing graphrag status

    # Assign the built graphrag details
    status_details["graphrag"] = graphrag_details

    # Re-evaluate overall_status based on all components
    # If any component status is 'error' or 'config_error', overall is 'error'
    # If any component status is 'degraded' or 'import_failed' AND no component is 'error' or 'config_error', overall is 'degraded'
    # Otherwise, overall is 'ok'
    component_statuses = [s.get("status") for s in status_details.values()]

    if any(s in ["error", "config_error"] for s in component_statuses):
         overall_status = "error"
    elif any(s in ["degraded", "import_failed", "unknown", "components_imported"] for s in component_statuses):
         overall_status = "degraded" # Any status other than 'loaded' or 'local_search_potentially_available'
    else:
         overall_status = "ok"


    # Determine message based on overall_status (remain the same)
    if overall_status == "error":
        overall_message = "一个或多个关键组件（模型或 GraphRAG 目录）加载失败，服务不可用。"
    elif overall_status == "degraded":
        overall_message = "部分组件（模型或 GraphRAG 导入）加载失败，服务可能降级。"
    else: # ok
        overall_message = "所有组件已成功加载并配置。"

    return {"status": overall_status, "message": overall_message, "details": status_details}


@app.route('/health', methods=['GET'])
def health_check():
    """Flask route for health check. Uses the helper function to get status."""
    # Use the helper function to get the status details
    status_data = get_loading_status_details()

    # Determine HTTP status code based on the overall status
    if status_data["status"] == "error":
        http_status_code = 503
    elif status_data["status"] == "degraded":
        http_status_code = 200 # Degraded is often still considered a successful HTTP response
    else: # ok
        http_status_code = 200

    # Return the status data as JSON response
    # Use Flask's app context to make jsonify work
    with app.app_context(): # <<< Activate app context
         return jsonify(status_data), http_status_code


@app.route('/generate', methods=['POST'])
def generate_text():
    """Handle Qwen model generation requests"""
    global tokenizer_definition, model_definition, tokenizer_calculation, model_calculation
    try:
        data = request.json
        prompt = data.get('prompt')
        question_type = data.get('question_type') # 'definition' or 'calculation'

        if not prompt:
            logger.warning("/generate: Missing 'prompt' field.")
            return jsonify({"error": "Request body missing 'prompt' field."}), 400

        if not question_type:
            logger.warning("/generate: Missing 'question_type' field.")
            return jsonify({"error": "Request body missing 'question_type' field ('definition' or 'calculation')."}), 400

        logger.info(f"/generate: Received request, type: '{question_type}', Prompt: '{prompt[:100]}...'")

        selected_tokenizer = None
        selected_model = None
        model_name_for_log = "Unknown"

        if question_type == "definition":
            selected_tokenizer = tokenizer_definition
            selected_model = model_definition
            model_name_for_log = "Qwen Definition Model" # Updated Name
        elif question_type == "calculation":
            selected_tokenizer = tokenizer_calculation
            selected_model = model_calculation
            model_name_for_log = "Qwen Calculation Model" # Updated Name
        else:
            logger.warning(f"/generate: Invalid 'question_type': {question_type}.")
            return jsonify({"error": f"Invalid 'question_type': {question_type}. Use 'definition' or 'calculation'."}), 400

        if selected_tokenizer is None or selected_model is None:
            logger.error(f"/generate: Requested '{model_name_for_log}' or its Tokenizer failed to load or initialize.")
            return jsonify({"error": f"Model '{model_name_for_log}' currently unavailable, please try again later or contact admin."}), 503 # Service Unavailable

        logger.info(f"/generate: Request will be handled by '{model_name_for_log}'.")

        # Determine target device for inputs.
        _, _, target_device = determine_device_and_dtype()

        inputs = selected_tokenizer(prompt, return_tensors="pt").to(target_device)
        logger.info(f"/generate: Moving input tensors to device: {inputs['input_ids'].device}")


        outputs = selected_model.generate(
            **inputs,
            max_new_tokens=2048,
            # temperature=0.7, # Adjust as needed
            # top_p=0.9,
            # do_sample=True
             pad_token_id=selected_tokenizer.eos_token_id # Good practice for generation
        )
        # Decode, removing the input prompt part from the output
        decoded_output = selected_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Simple slicing: find where the prompt ends in the decoded output and take the rest
        if prompt in decoded_output:
             generated_text = decoded_output[decoded_output.index(prompt) + len(prompt):].strip()
        else:
             generated_text = decoded_output.strip()


        logger.info(f"/generate: Text generation completed. Output length: {len(generated_text)}")
        return jsonify({"response": generated_text})

    except Exception as e:
        logger.error(f"/generate endpoint encountered an error: {e}", exc_info=True)
        return jsonify({"error": f"Server internal error: {str(e)}"}), 500


@app.route('/query_graphrag', methods=['POST'])
def query_graphrag():
    """Handle GraphRAG query requests (Local Search only)"""
    # Note: Flask routes are synchronous by default. asyncio.run is used to call async GraphRAG functions.

    # Check GraphRAG availability using the helper function
    status_data = get_loading_status_details()
    # --- ADD DEBUG LOG HERE ---
    logger.info(f"DEBUG: Status details from get_loading_status_details() in /query_graphrag: {status_data}")
    # --- CORRECTED ACCESS ---
    graphrag_component_status = status_data.get("details", {}).get("graphrag", {}).get("status", 'N/A') # <<< CORRECTED ACCESS
    logger.info(f"DEBUG: graphrag status is: {graphrag_component_status}")
    # --- END DEBUG LOG ---


    if graphrag_component_status not in ["local_search_potentially_available"]: # Use the correctly extracted status
         # We check specifically if the graphrag component is marked as available for local search
         logger.warning(f"/query_graphrag: GraphRAG Local Search is not available (status: {graphrag_component_status}).")
         return jsonify({"error": "GraphRAG 功能当前不可用，因为其组件未能正确加载或配置。"}), 503 # Service Unavailable


    try:
        data = request.json
        query = data.get('query')
        # method is now hardcoded 'local' on the frontend,
        # we still get it from request but will enforce 'local'
        method = data.get('method', 'local') # Default to local if not provided (less likely with frontend change)


        if not query:
            logger.warning("/query_graphrag: Missing 'query' field.")
            return jsonify({"error": "Request body missing 'query' field."}), 400

        # --- ENFORCE LOCAL METHOD ONLY ---
        if method != 'local':
            logger.warning(f"/query_graphrag: Invalid method '{method}'. Only 'local' is supported.")
            return jsonify({"error": "仅支持 'local' 查询方法。"}), 400
        # --- END ENFORCE ---

        logger.info(f"/query_graphrag: Received request, method: '{method}', Query: '{query[:100]}...'")

        result_text = ""
        context_data = {} # GraphRAG functions return context data too

        # Use a nested async function to perform the async calls
        async def perform_graphrag_query_async():
            # 1. Load GraphRAG configuration (This is NOT async)
            logger.info(f"/query_graphrag: Loading GraphRAG config from {GRAPH_RAG_ROOT_DIR}")
            # Convert the string path to a Path object before passing it
            graphrag_root_path = Path(GRAPH_RAG_ROOT_DIR) # Convert to Path object

            try:
                config = load_config(root_dir=graphrag_root_path, config_filepath=None, cli_overrides={}) # Use Path object
                logger.info("/query_graphrag: GraphRAG config loaded.")
            except Exception as e:
                 # Capture specific config loading errors
                 logger.error(f"GraphRAG config loading failed: {e}", exc_info=True)
                 raise Exception(f"GraphRAG 配置加载失败: {str(e)}. 请检查 '{GRAPH_RAG_ROOT_DIR}' 目录下的 rag.yaml 文件。") from e # Re-raise with context


            # 2. Load necessary dataframes - Hardcoded for Local Search only
            # required_outputs and optional_outputs are now handled inside load_graphrag_local_dataframes
            try:
                # Use the specific helper for Local Search data loading
                dataframe_dict = await load_graphrag_local_dataframes( # AWAIT IS INSIDE ASYNC FUNCTION
                    root_dir=graphrag_root_path, # Pass the Path object now
                    config=config,
                )
                logger.info("/query_graphrag: GraphRAG dataframes loaded for Local Search.")

                # The missing dataframe check is integrated inside load_graphrag_local_dataframes

            except Exception as load_error:
                 # Catch loading errors from load_graphrag_local_dataframes
                 logger.error(f"GraphRAG Local Search dataframes loading failed: {load_error}", exc_info=True)
                 raise Exception(f"GraphRAG Local Search 数据文件加载失败: {str(load_error)}") from load_error # Re-raise with context


            # 3. Call the Local Search API function asynchronously
            # Determine community_level for Local Search
            community_level_to_pass = getattr(getattr(config, 'community_report_generation', None), 'level', 2) # Default to 2
            if community_level_to_pass is None: # Double check default setting
                 community_level_to_pass = 2 # Ensure it's not None if getattr returned None unexpectedly
                 logger.warning(f"GraphRAG config.community_report_generation.level is None, using default level {community_level_to_pass} for local search.")

            # Safely access response_type from config, provide a default if missing
            response_type = getattr(config, 'response_type', 'text') # Using getattr here
            logger.info(f"/query_graphrag: Using response_type: {response_type}")

            streaming = False # Not implementing streaming via Flask for simplicity
            # Timeout for the API call itself (optional, but good practice)
            api_call_timeout = 60 # seconds, adjust as needed


            try: # Wrap API call in try block as well
                logger.info(f"/query_graphrag: Calling graphrag.api.local_search...")

                # Wrap the API call with a timeout
                response, context = await asyncio.wait_for(
                    local_search(
                        config=config,
                        entities=dataframe_dict.get("entities"),
                        communities=dataframe_dict.get("communities"),
                        community_reports=dataframe_dict.get("community_reports"),
                        text_units=dataframe_dict.get("text_units"),
                        relationships=dataframe_dict.get("relationships"),
                        covariates=dataframe_dict.get("covariates"),
                        community_level=community_level_to_pass, # Pass community_level
                        response_type=response_type, # Use the determined response_type
                        query=query,
                        callbacks=None # No callbacks for simple test
                    ), timeout=api_call_timeout # Apply timeout here
                )
                logger.info("/query_graphrag: graphrag.api.local_search completed successfully.")


            except asyncio.TimeoutError:
                 # Capture timeout specifically
                 logger.error(f"GraphRAG Local Search API call timed out after {api_call_timeout} seconds.")
                 raise asyncio.TimeoutError(f"GraphRAG Local Search API 调用超时 ({api_call_timeout}秒)") # Re-raise as specific TimeoutError
            except Exception as api_call_error:
                # Capture other errors during the GraphRAG API call itself
                logger.error(f"GraphRAG Local Search API call failed: {api_call_error}", exc_info=True)
                raise Exception(f"GraphRAG Local Search API 调用失败: {str(api_call_error)}") from api_call_error # Re-raise with context


            return response, context # Success case

        # --- End of nested async function definition ---

        # Now, run the nested async function using asyncio.run()
        # Add overall timeout for the entire GraphRAG query process
        overall_timeout = 120 # seconds, adjust as needed (should be > api_call_timeout)
        try:
            result_text, context_data = asyncio.run(asyncio.wait_for(perform_graphrag_query_async(), timeout=overall_timeout))

            # GraphRAG API function returns text response.
            if not result_text or result_text.strip() == "":
                 result_text = "GraphRAG 未能找到相关信息或生成回答。"
                 logger.warning(f"/query_graphrag: 查询 '{query}' ({method}) 返回空结果。")

        # Catch specific GraphRAG related errors propagated from async function
        except FileNotFoundError as fnf_error:
            logger.error(f"/query_graphrag: 捕获到文件缺失错误: {fnf_error}")
            return jsonify({"error": f"GraphRAG 数据文件缺失: {str(fnf_error)}。请确认 '{GRAPH_RAG_ROOT_DIR}' 目录已包含 GraphRAG 索引构建后的必要输出文件。"}), 500

        except asyncio.TimeoutError as timeout_error_propagated:
            logger.error(f"/query_graphrag: 捕获到 GraphRAG 查询超时错误: {timeout_error_propagated}")
            return jsonify({"error": f"GraphRAG 查询超时: {str(timeout_error_propagated)}。请尝试更精简的查询。"}), 504 # Gateway Timeout

        except Exception as graphrag_error_propagated:
            # This will now catch config loading errors, data loading errors, and API call errors
            logger.error(f"/query_graphrag: 捕获到 GraphRAG 执行错误: {graphrag_error_propagated}")
            return jsonify({"error": f"执行 GraphRAG 查询失败: {str(graphrag_error_propagated)}. 详细信息请查看后端日志。"}), 500


        logger.info(f"/query_graphrag: GraphRAG Query completed. Result length: {len(result_text)}")
        return jsonify({"response": result_text})

    except Exception as e:
        # Catch any errors before or during initial request processing (e.g., JSON parsing, method check)
        logger.error(f"/query_graphrag endpoint encountered a non-GraphRAG execution error: {e}", exc_info=True)
        return jsonify({"error": f"服务器内部错误 (处理 GraphRAG 请求时): {str(e)}"}), 500




if __name__ == '__main__':
    try:
        logger.info("--- Application starting up ---")
        # No longer calling health_check() here
        load_all_models_and_tokenizers()
        logger.info("--- Models and components load attempts completed ---")

        # Get the final status after loading using the helper function
        status_at_start = get_loading_status_details() # <<< Use the helper function directly
        overall_status_at_start = status_at_start.get("status", "unknown")


        if overall_status_at_start == "error":
             logger.critical(f"Critical components failed to load ({overall_status_at_start} status, see logs for details), application may not function correctly or will exit.")
             # Optionally exit here if critical components are missing
             # import sys
             # sys.exit(1)
        elif overall_status_at_start == "degraded":
             logger.warning(f"Partial components failed to load ({overall_status_at_start} status, see logs for details), service might be degraded.")
        else:
             logger.info(f"All components loaded successfully ({overall_status_at_start} status).")


        logger.info("Flask development server ready to start, listening on port 6006.")
        logger.info("Note: For production use, use a WSGI server like Gunicorn or Waitress.")
        # Ensure GRAPH_RAG_ROOT_DIR is a valid relative or absolute path before starting the server
        # This check is already done in load_all_models_and_tokenizers, but a final check doesn't hurt
        # Check again here based on final status just before starting the server
        final_status_check = get_loading_status_details()
        if final_status_check.get("graphrag", {}).get("status") == "config_error":
             # Config_error happens when imported but directory is missing
             logger.critical(f"GraphRAG root directory '{GRAPH_RAG_ROOT_DIR}' is missing or invalid. Cannot start application with full functionality.")
             # Decide whether to exit here or allow degraded startup
             # For now, log critical and proceed, but service will return 503 for GraphRAG
             # import sys
             # sys.exit(1) # Uncomment to force exit

        app.run(host='0.0.0.0', port=6006, debug=False) # In production, set debug=False

    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)