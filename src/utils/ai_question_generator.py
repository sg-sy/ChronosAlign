"""
AI-powered question generator using Anthropic Claude API, OpenAI API, or Hugging Face models.
"""

import os
import json
from typing import Dict, List, Any, Optional, Literal
from anthropic import Anthropic
from openai import OpenAI

from .logger import LoggerMixin


class AIQuestionGenerator(LoggerMixin):
    """
    Generate question-answer sets from content using Anthropic Claude API, OpenAI API, or Hugging Face models.
    """

    def __init__(
        self,
        provider: Literal["anthropic", "openai", "huggingface"] = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the AI question generator.

        Args:
            provider: LLM provider ("anthropic", "openai", or "huggingface")
            api_key: API key for the specified provider. If None, will try to get from environment
            model: Model name to use. If None, will use default for the provider
            device: Device to run Hugging Face models on ("cpu" or "cuda")
            cache_dir: Cache directory for Hugging Face models (optional)
        """
        super().__init__()

        self.provider = provider
        self.device = device
        self.cache_dir = cache_dir

        # Get API key based on provider
        if api_key:
            self.api_key = api_key
        else:
            if provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
                    )
            elif provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
                    )
            elif provider == "huggingface":
                # No API key needed for Hugging Face models
                self.api_key = None
            else:
                raise ValueError(
                    f"Unsupported provider: {provider}. Supported providers are 'anthropic', 'openai', and 'huggingface'"
                )

        # Initialize client based on provider
        if provider == "anthropic":
            self.client = Anthropic(api_key=self.api_key)
            self.model = model or "claude-sonnet-4-20250514"
        elif provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
            self.model = model or "gpt-4o"
        elif provider == "huggingface":
            # For Hugging Face, we'll load the model when needed
            self.client = None
            self.model = model or "gpt2"
            # Store cache directory for Hugging Face models
            self.cache_dir = cache_dir
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers are 'anthropic', 'openai', and 'huggingface'"
            )

    def generate_qa_set(
        self,
        content: str,
        system_prompt: str,
        user_prompt: str,
        num_questions: Optional[int] = None,
        max_tokens: int = 4000,
        wikipedia_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a set of question-answer pairs from content using the specified LLM provider.

        Args:
            content: Input content to generate questions from
            system_prompt: System prompt for the AI
            user_prompt: User prompt template (should include {content} placeholder)
            num_questions: Number of questions to generate (None for unlimited)
            max_tokens: Maximum tokens for the response
            wikipedia_url: Optional Wikipedia source URL

        Returns:
            List of question-answer dictionaries
        """
        try:
            # Format the user prompt with content
            formatted_user_prompt = user_prompt.format(content=content)

            # Make API call based on provider
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": formatted_user_prompt}],
                )
                response_text = response.content[0].text
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": formatted_user_prompt},
                    ],
                )
                response_text = response.choices[0].message.content
            elif self.provider == "huggingface":
                # Use Hugging Face pipeline
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                import torch

                full_prompt = f"{system_prompt}\n{formatted_user_prompt}"

                # Load model and tokenizer with cache directory if specified
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model, cache_dir=self.cache_dir, device_map="auto"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model, cache_dir=self.cache_dir, device_map="auto"
                )
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto",
                )

                # Generate output
                outputs = pipe(full_prompt, max_new_tokens=max_tokens, truncation=True)
                response_text = (
                    outputs[0]["generated_text"]
                    if outputs and "generated_text" in outputs[0]
                    else outputs[0]["text"]
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            # Parse the response
            qa_pairs = self._parse_qa_response(response_text)

            # Add Wikipedia source URL to each QA pair if provided
            if wikipedia_url:
                for qa_pair in qa_pairs:
                    qa_pair["wikipedia_source_url"] = wikipedia_url

            # Limit to requested number of questions only if specified
            if num_questions is not None and len(qa_pairs) > num_questions:
                qa_pairs = qa_pairs[:num_questions]

            self.log_info(
                f"Generated {len(qa_pairs)} question-answer pairs using {self.provider}"
            )
            return qa_pairs

        except Exception as e:
            self.log_error(f"Error generating QA set: {e}")
            return []

    def generate_qa_from_wikipedia(
        self,
        wikipedia_url: str,
        system_prompt: str,
        user_prompt: str,
        num_questions: Optional[int] = None,
        max_tokens: int = 4000,
    ) -> List[Dict[str, Any]]:
        """
        Generate question-answer pairs from a Wikipedia page.

        Args:
            wikipedia_url: Wikipedia page URL
            system_prompt: System prompt for the AI
            user_prompt: User prompt template (should include {content} placeholder)
            num_questions: Number of questions to generate (None for unlimited)
            max_tokens: Maximum tokens for the response

        Returns:
            List of question-answer dictionaries
        """
        try:
            # Extract Wikipedia content
            from .wikipedia_extractor import WikipediaExtractor

            extractor = WikipediaExtractor()

            result = extractor.extract_wikipedia_content(wikipedia_url)

            if not result["success"]:
                self.log_error(
                    f"Failed to extract Wikipedia content: {result.get('error', 'Unknown error')}"
                )
                return []

            # Generate QA pairs from the extracted content
            return self.generate_qa_set(
                content=result["content"],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                num_questions=num_questions,
                max_tokens=max_tokens,
                wikipedia_url=wikipedia_url,
            )

        except Exception as e:
            self.log_error(f"Error generating QA from Wikipedia: {e}")
            return []

    def generate_qa_batch(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        num_questions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate question-answer pairs from multiple content items using Anthropic's batch processing API.

        This method follows the Anthropic batch processing specification:
        https://docs.anthropic.com/en/docs/build-with-claude/batch-processing

        Args:
            contents: List of dictionaries containing 'content' and 'custom_id' (source URL) keys
            system_prompt: System prompt for the AI
            user_prompt: User prompt template (should include {content} placeholder)
            max_tokens: Maximum tokens for the response
            num_questions: Number of questions to generate per content (None for unlimited)

        Returns:
            Dictionary containing batch information and results
        """
        if self.provider not in ["anthropic", "openai"]:
            raise ValueError(
                f"Batch processing is only supported for 'anthropic' and 'openai' providers, not '{self.provider}'"
            )

        try:
            if self.provider == "anthropic":
                return self._create_anthropic_batch(
                    contents, system_prompt, user_prompt, max_tokens
                )
            elif self.provider == "openai":
                return self._create_openai_batch(
                    contents, system_prompt, user_prompt, max_tokens
                )
        except Exception as e:
            self.log_error(f"Error creating batch: {e}")
            raise

        except Exception as e:
            self.log_error(f"Error creating batch: {e}")
            raise

    def _create_anthropic_batch(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """
        Create a batch using Anthropic's batch API.

        Args:
            contents: List of content items with custom_ids
            system_prompt: System prompt for the AI
            user_prompt: User prompt template
            max_tokens: Maximum tokens for the response

        Returns:
            Dictionary containing batch information
        """
        # Import required classes for batch API
        from anthropic.types.message_create_params import (
            MessageCreateParamsNonStreaming,
        )
        from anthropic.types.messages.batch_create_params import Request

        # Prepare batch requests according to Anthropic's specification
        batch_requests = []
        custom_ids = []

        for content_item in contents:
            content = content_item["content"]
            custom_id = content_item.get("custom_id", "")

            if not custom_id:
                self.log_warning(f"No custom_id provided for content, using default")
                custom_id = f"content_{len(batch_requests)}"

            # Format the user prompt with content
            formatted_user_prompt = user_prompt.format(content=content)

            # Create request using the proper Request and MessageCreateParamsNonStreaming classes
            request = Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": formatted_user_prompt}],
                ),
            )

            batch_requests.append(request)
            custom_ids.append(custom_id)

        self.log_info(f"Creating Anthropic batch with {len(batch_requests)} requests")

        # Create the batch using Anthropic's batch API
        try:
            batch_response = self.client.messages.batches.create(
                requests=batch_requests
            )
        except AttributeError as e:
            raise ValueError(
                f"Batch API not available in current Anthropic SDK version. "
                f"Please update with: pip install --upgrade anthropic. "
                f"Error: {e}"
            )

        self.log_info(f"Anthropic batch created with ID: {batch_response.id}")
        self.log_info(f"Processing status: {batch_response.processing_status}")

        # Return batch information for tracking
        return {
            "batch_id": batch_response.id,
            "processing_status": batch_response.processing_status,
            "request_counts": batch_response.request_counts,
            "created_at": batch_response.created_at,
            "custom_ids": custom_ids,
        }

    def _create_anthropic_batch_with_questions(
        self,
        contents: List[Dict[str, Any]],
        questions: List[str],
        system_prompt: str,
        user_prompt_update_questions: str,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """
        Create a batch using Anthropic's batch API, providing both contents and related questions.
        Each batch item will process a question update prompt with the corresponding content and question.

        Args:
            contents: List of content items (dicts with at least 'content', optionally 'custom_id')
            questions: List of question strings matching the length of contents
            system_prompt: System prompt for the AI
            user_prompt_update_questions: User prompt template with placeholders for 'question' and 'content'
            max_tokens: Maximum tokens for the response

        Returns:
            Dictionary containing batch information
        """
        if len(contents) != len(questions):
            raise ValueError("Length of contents and questions must match.")

        batch_requests = []
        custom_ids = []

        for i, (content_item, question) in enumerate(zip(contents, questions)):
            content = content_item["content"]
            custom_id = content_item.get("custom_id", "")
            if not custom_id:
                self.log_warning(f"No custom_id provided for content at index {i}, using default")
                custom_id = f"content_{i}"

            formatted_user_prompt = user_prompt_update_questions.format(
                question=question, content=content
            )

            request = Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": formatted_user_prompt}],
                ),
            )
            batch_requests.append(request)
            custom_ids.append(custom_id)

        self.log_info(f"Creating Anthropic batch with {len(batch_requests)} requests (questions aware)")

        # Create the batch using Anthropic's batch API
        try:
            batch_response = self.client.messages.batches.create(
                requests=batch_requests
            )
        except AttributeError as e:
            raise ValueError(
                f"Batch API not available in current Anthropic SDK version. "
                f"Please update with: pip install --upgrade anthropic. "
                f"Error: {e}"
            )

        self.log_info(f"Anthropic batch (questions aware) created with ID: {batch_response.id}")
        self.log_info(f"Processing status: {batch_response.processing_status}")

        return {
            "batch_id": batch_response.id,
            "processing_status": batch_response.processing_status,
            "request_counts": batch_response.request_counts,
            "created_at": batch_response.created_at,
            "custom_ids": custom_ids,
        }

    def _create_openai_batch(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """
        Create a batch using OpenAI's batch API.

        Args:
            contents: List of content items with custom_ids
            system_prompt: System prompt for the AI
            user_prompt: User prompt template
            max_tokens: Maximum tokens for the response

        Returns:
            Dictionary containing batch information
        """
        import tempfile
        import json

        # Prepare batch requests according to OpenAI's specification
        batch_requests = []
        custom_ids = []

        for content_item in contents:
            content = content_item["content"]
            custom_id = content_item.get("custom_id", "")

            if not custom_id:
                self.log_warning(f"No custom_id provided for content, using default")
                custom_id = f"content_{len(batch_requests)}"

            # Format the user prompt with content
            formatted_user_prompt = user_prompt.format(content=content)

            # Create request according to OpenAI batch API specification
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "max_completion_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": formatted_user_prompt},
                    ],
                },
            }

            batch_requests.append(request)
            custom_ids.append(custom_id)

        self.log_info(f"Creating OpenAI batch with {len(batch_requests)} requests")

        # Create a temporary JSONL file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as temp_file:
            for request in batch_requests:
                temp_file.write(json.dumps(request) + "\n")
            temp_file_path = temp_file.name

        try:
            # Upload the JSONL file to OpenAI
            with open(temp_file_path, "rb") as file:
                upload_response = self.client.files.create(file=file, purpose="batch")

            self.log_info(f"Uploaded JSONL file with ID: {upload_response.id}")

            # Create the batch using the uploaded file
            batch_response = self.client.batches.create(
                input_file_id=upload_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": "QA Generation Batch"},
            )

            self.log_info(f"OpenAI batch created with ID: {batch_response.id}")
            self.log_info(f"Processing status: {batch_response.status}")

            # Return batch information for tracking
            return {
                "batch_id": batch_response.id,
                "processing_status": batch_response.status,
                "request_counts": {"total": len(batch_requests)},
                "created_at": batch_response.created_at,
                "custom_ids": custom_ids,
                "input_file_id": upload_response.id,
            }

        except Exception as e:
            raise ValueError(
                f"OpenAI batch API error. Please check your API key and batch API access. "
                f"Error: {e}"
            )
        finally:
            # Clean up the temporary file
            try:
                import os

                os.unlink(temp_file_path)
            except:
                pass

    def generate_qa_batch_with_questions(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        num_questions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate question-answer pairs from multiple content items with questions using Anthropic's batch processing API.

        This method supports user prompts with both {content} and {questions} placeholders.
        This method follows the Anthropic batch processing specification:
        https://docs.anthropic.com/en/docs/build-with-claude/batch-processing

        Args:
            contents: List of dictionaries containing 'content', 'questions' (or 'question'), and 'custom_id' keys
            system_prompt: System prompt for the AI
            user_prompt: User prompt template (should include {content} and {questions} or {question} placeholders)
            max_tokens: Maximum tokens for the response
            num_questions: Number of questions to generate per content (None for unlimited)

        Returns:
            Dictionary containing batch information and results
        """
        if self.provider != "anthropic":
            raise ValueError(
                f"Batch processing with questions is only supported for 'anthropic' provider, not '{self.provider}'"
            )

        try:
            return self._create_anthropic_batch_with_questions(
                contents, system_prompt, user_prompt, max_tokens
            )
        except Exception as e:
            self.log_error(f"Error creating batch with questions: {e}")
            raise

    def _create_anthropic_batch_with_questions(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """
        Create a batch using Anthropic's batch API with support for both content and questions substitution.

        Args:
            contents: List of content items with custom_ids, content, and questions/question fields
            system_prompt: System prompt for the AI
            user_prompt: User prompt template (should include {content} and {questions} or {question} placeholders)
            max_tokens: Maximum tokens for the response

        Returns:
            Dictionary containing batch information
        """
        # Import required classes for batch API
        from anthropic.types.message_create_params import (
            MessageCreateParamsNonStreaming,
        )
        from anthropic.types.messages.batch_create_params import Request

        # Prepare batch requests according to Anthropic's specification
        batch_requests = []
        custom_ids = []

        for content_item in contents:
            content = content_item["content"]
            custom_id = content_item.get("custom_id", "")

            # Get questions - support both 'questions' and 'question' keys
            questions = content_item.get("questions", content_item.get("question", ""))

            # Convert questions to string if it's a list
            if isinstance(questions, list):
                questions_str = "\n".join(str(q) for q in questions)
            elif isinstance(questions, dict):
                # If it's a dict, try to format it nicely
                questions_str = json.dumps(questions, indent=2)
            else:
                questions_str = str(questions) if questions else ""

            if not custom_id:
                self.log_warning(f"No custom_id provided for content, using default")
                custom_id = f"content_{len(batch_requests)}"

            # Format the user prompt with both content and questions
            # Support both {questions} and {question} placeholders
            # Build format kwargs based on what placeholders exist in the prompt
            format_kwargs = {}

            # Check which placeholders are in the prompt
            has_content = "{content}" in user_prompt
            has_questions = "{questions}" in user_prompt
            has_question = "{question}" in user_prompt

            if has_content:
                format_kwargs["content"] = content
            if has_questions:
                format_kwargs["questions"] = questions_str
            if has_question:
                format_kwargs["question"] = questions_str

            # Format the prompt with available kwargs
            formatted_user_prompt = user_prompt.format(**format_kwargs)

            # Create request using the proper Request and MessageCreateParamsNonStreaming classes
            request = Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": formatted_user_prompt}],
                ),
            )

            batch_requests.append(request)
            custom_ids.append(custom_id)

        self.log_info(
            f"Creating Anthropic batch with {len(batch_requests)} requests (with questions support)"
        )

        # Create the batch using Anthropic's batch API
        try:
            batch_response = self.client.messages.batches.create(
                requests=batch_requests
            )
        except AttributeError as e:
            raise ValueError(
                f"Batch API not available in current Anthropic SDK version. "
                f"Please update with: pip install --upgrade anthropic. "
                f"Error: {e}"
            )

        self.log_info(f"Anthropic batch created with ID: {batch_response.id}")
        self.log_info(f"Processing status: {batch_response.processing_status}")

        # Return batch information for tracking
        return {
            "batch_id": batch_response.id,
            "processing_status": batch_response.processing_status,
            "request_counts": batch_response.request_counts,
            "created_at": batch_response.created_at,
            "custom_ids": custom_ids,
        }

    def _get_anthropic_batch_results(self, batch_id: str) -> Dict[str, Any]:
        """
        Retrieve results from an Anthropic batch processing job.

        Args:
            batch_id: The ID of the batch to retrieve results for

        Returns:
            Dictionary containing batch results and status
        """
        # Retrieve batch results
        batch = self.client.messages.batches.retrieve(batch_id)

        self.log_info(f"Anthropic batch {batch_id} status: {batch.processing_status}")

        # If batch is still processing, return current status
        if batch.processing_status != "completed":
            return {
                "batch_id": batch.id,
                "processing_status": batch.processing_status,
                "request_counts": batch.request_counts,
                "created_at": batch.created_at,
                "ended_at": batch.ended_at,
                "results": None,
            }

        # Process completed results
        all_qa_pairs = []
        successful_requests = 0
        failed_requests = 0

        for batch_result in batch.batch:
            custom_id = batch_result.custom_id

            if batch_result.status == "success":
                # Parse the response
                response_text = batch_result.response.content[0].text
                qa_pairs = self._parse_qa_response(response_text)

                # Add source information to each QA pair
                for qa_pair in qa_pairs:
                    qa_pair["source_custom_id"] = custom_id
                    qa_pair["source_batch_id"] = batch_id

                all_qa_pairs.extend(qa_pairs)
                successful_requests += 1

                self.log_info(
                    f"Successfully processed {custom_id}: generated {len(qa_pairs)} QA pairs"
                )

            else:
                # Handle failed requests
                error_msg = (
                    batch_result.error.message
                    if batch_result.error
                    else "Unknown error"
                )
                self.log_error(f"Failed to process {custom_id}: {error_msg}")
                failed_requests += 1

        self.log_info(
            f"Anthropic batch {batch_id} completed: {successful_requests} successful, {failed_requests} failed"
        )

        return {
            "batch_id": batch.id,
            "processing_status": batch.processing_status,
            "request_counts": batch.request_counts,
            "created_at": batch.created_at,
            "ended_at": batch.ended_at,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "total_qa_pairs": len(all_qa_pairs),
            "qa_pairs": all_qa_pairs,
        }

    def _get_openai_batch_results(self, batch_id: str) -> Dict[str, Any]:
        """
        Retrieve results from an OpenAI batch processing job.

        Args:
            batch_id: The ID of the batch to retrieve results for

        Returns:
            Dictionary containing batch results and status
        """
        import json
        import tempfile

        # Retrieve batch results
        batch = self.client.batches.retrieve(batch_id)

        self.log_info(f"OpenAI batch {batch_id} status: {batch.status}")

        # If batch is still processing, return current status
        if batch.status not in ["completed", "failed", "expired"]:
            return {
                "batch_id": batch.id,
                "processing_status": batch.status,
                "request_counts": {
                    "total": (
                        batch.request_counts.total
                        if hasattr(batch, "request_counts")
                        else 0
                    )
                },
                "created_at": batch.created_at,
                "ended_at": batch.ended_at if hasattr(batch, "ended_at") else None,
                "results": None,
            }

        # Process completed results
        all_qa_pairs = []
        successful_requests = 0
        failed_requests = 0

        # Download and parse the output file if available
        if hasattr(batch, "output_file_id") and batch.output_file_id:
            self.log_info(
                f"OpenAI batch {batch_id} completed with output file: {batch.output_file_id}"
            )

            try:
                # Download the output file
                output_file = self.client.files.retrieve(batch.output_file_id)

                # Download the file content
                file_content = self.client.files.content(batch.output_file_id)

                # Parse the JSONL output file
                for line in file_content.iter_lines():
                    if line:
                        try:
                            result = json.loads(line.decode("utf-8"))

                            # Check if this was a successful request
                            if result.get("status") == "success":
                                custom_id = result.get("custom_id", "")
                                response = result.get("response", {})

                                # Extract the response content
                                if (
                                    "choices" in response
                                    and len(response["choices"]) > 0
                                ):
                                    response_text = (
                                        response["choices"][0]
                                        .get("message", {})
                                        .get("content", "")
                                    )

                                    if response_text:
                                        # Parse the QA pairs from the response
                                        qa_pairs = self._parse_qa_response(
                                            response_text
                                        )

                                        # Add source information to each QA pair
                                        for qa_pair in qa_pairs:
                                            qa_pair["source_custom_id"] = custom_id
                                            qa_pair["source_batch_id"] = batch_id

                                        all_qa_pairs.extend(qa_pairs)
                                        successful_requests += 1

                                        self.log_info(
                                            f"Successfully processed {custom_id}: generated {len(qa_pairs)} QA pairs"
                                        )
                                    else:
                                        self.log_warning(
                                            f"No response content for {custom_id}"
                                        )
                                        failed_requests += 1
                                else:
                                    self.log_warning(
                                        f"No valid choices in response for {custom_id}"
                                    )
                                    failed_requests += 1
                            else:
                                # Handle failed requests
                                error_msg = result.get("error", {}).get(
                                    "message", "Unknown error"
                                )
                                custom_id = result.get("custom_id", "unknown")
                                self.log_error(
                                    f"Failed to process {custom_id}: {error_msg}"
                                )
                                failed_requests += 1

                        except json.JSONDecodeError as e:
                            self.log_error(f"Error parsing batch result line: {e}")
                            failed_requests += 1
                        except Exception as e:
                            self.log_error(f"Error processing batch result: {e}")
                            failed_requests += 1

            except Exception as e:
                self.log_error(f"Error downloading or parsing output file: {e}")
                failed_requests += 1
        else:
            self.log_warning(f"No output file available for batch {batch_id}")

        self.log_info(
            f"OpenAI batch {batch_id} completed: {successful_requests} successful, {failed_requests} failed"
        )

        return {
            "batch_id": batch.id,
            "processing_status": batch.status,
            "request_counts": {
                "total": (
                    batch.request_counts.total
                    if hasattr(batch, "request_counts")
                    else 0
                )
            },
            "created_at": batch.created_at,
            "ended_at": batch.ended_at if hasattr(batch, "ended_at") else None,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "total_qa_pairs": len(all_qa_pairs),
            "qa_pairs": all_qa_pairs,
            "output_file_id": (
                batch.output_file_id if hasattr(batch, "output_file_id") else None
            ),
        }

    def get_batch_results(self, batch_id: str) -> Dict[str, Any]:
        """
        Retrieve results from a batch processing job.

        Args:
            batch_id: The ID of the batch to retrieve results for

        Returns:
            Dictionary containing batch results and status
        """
        if self.provider not in ["anthropic", "openai"]:
            raise ValueError(
                f"Batch processing is only supported for 'anthropic' and 'openai' providers, not '{self.provider}'"
            )

        try:
            if self.provider == "anthropic":
                return self._get_anthropic_batch_results(batch_id)
            elif self.provider == "openai":
                return self._get_openai_batch_results(batch_id)

        except Exception as e:
            self.log_error(f"Error retrieving batch results: {e}")
            raise

    def wait_for_batch_completion(
        self, batch_id: str, poll_interval: int = 30, timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Wait for a batch to complete and return results.

        Args:
            batch_id: The ID of the batch to wait for
            poll_interval: Seconds to wait between status checks
            timeout: Maximum seconds to wait before timing out

        Returns:
            Dictionary containing batch results
        """
        import time

        start_time = time.time()

        while True:
            # Check if we've exceeded timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Batch {batch_id} did not complete within {timeout} seconds"
                )

            # Get current batch status
            batch_info = self.get_batch_results(batch_id)

            # Handle different status names for different providers
            status = batch_info["processing_status"]
            if status == "completed":
                self.log_info(f"Batch {batch_id} completed successfully")
                return batch_info
            elif status in ["failed", "cancelled"]:
                raise Exception(f"Batch {batch_id} failed during processing")
            elif status == "canceled":
                raise Exception(f"Batch {batch_id} was canceled")
            elif status in ["expired", "timed_out"]:
                raise Exception(f"Batch {batch_id} expired before completion")

            # Still processing, wait before checking again
            self.log_info(
                f"Batch {batch_id} still processing, waiting {poll_interval} seconds..."
            )
            time.sleep(poll_interval)

    def _parse_qa_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse the AI response into structured question-answer pairs with metadata.

        Args:
            response_text: Raw response text from the AI

        Returns:
            List of dictionaries containing question, answer and metadata fields
        """
        import re

        qa_pairs = []

        try:
            # Try to extract and parse JSON array/object from anywhere in the response
            json_match = re.search(r"(\[.*?\]|\{.*?\})", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list):
                        qa_pairs = parsed
                    elif isinstance(parsed, dict):
                        if "questions" in parsed:
                            qa_pairs = parsed["questions"]
                        elif all(k in parsed for k in ["question", "answer"]):
                            qa_pairs = [parsed]
                except json.JSONDecodeError:
                    pass

            # If JSON parsing failed, try parsing from start of response
            if not qa_pairs and (
                response_text.strip().startswith("{")
                or response_text.strip().startswith("[")
            ):
                try:
                    parsed = json.loads(response_text)
                    if isinstance(parsed, list):
                        qa_pairs = parsed
                    elif isinstance(parsed, dict):
                        if "questions" in parsed:
                            qa_pairs = parsed["questions"]
                        elif all(k in parsed for k in ["question", "answer"]):
                            qa_pairs = [parsed]
                except json.JSONDecodeError:
                    pass

            # Try markdown and numbered formats if JSON parsing failed
            if not qa_pairs:
                qa_pairs = self._parse_markdown_qa(response_text)
            if not qa_pairs:
                qa_pairs = self._parse_numbered_qa(response_text)

            # Validate and structure the pairs
            cleaned_pairs = []
            for pair in qa_pairs:
                if isinstance(pair, dict) and "question" in pair and "answer" in pair:
                    cleaned_pair = {
                        "event_year": pair.get("event_year", ""),
                        "event_date": pair.get("event_date", ""),
                        "subject": pair.get("subject", ""),
                        "relationship": pair.get("relationship", ""),
                        "question": str(pair["question"]).strip(),
                        "answer": str(pair["answer"]).strip(),
                    }
                    cleaned_pairs.append(cleaned_pair)

            return cleaned_pairs

        except Exception as e:
            self.log_warning(f"Error parsing QA response: {e}")
            return []

    def _parse_markdown_qa(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse markdown-formatted question-answer pairs.

        Args:
            text: Markdown text

        Returns:
            List of question-answer dictionaries
        """
        qa_pairs = []
        lines = text.split("\n")

        current_question = None
        current_answer = []

        for line in lines:
            line = line.strip()

            # Check for question markers
            if (
                line.startswith("**Q:")
                or line.startswith("**Question:")
                or line.startswith("Q:")
                or line.startswith("Question:")
            ):
                # Save previous pair if exists
                if current_question and current_answer:
                    qa_pairs.append(
                        {
                            "question": current_question,
                            "answer": "\n".join(current_answer).strip(),
                            "metadata": {},
                        }
                    )

                # Start new question
                current_question = (
                    line.split(":", 1)[1].strip() if ":" in line else line
                )
                current_answer = []

            # Check for answer markers
            elif (
                line.startswith("**A:")
                or line.startswith("**Answer:")
                or line.startswith("A:")
                or line.startswith("Answer:")
            ):
                # Start collecting answer
                answer_text = line.split(":", 1)[1].strip() if ":" in line else line
                if answer_text:
                    current_answer.append(answer_text)

            # Collect answer lines
            elif current_question and line and not line.startswith("---"):
                current_answer.append(line)

        # Add the last pair
        if current_question and current_answer:
            qa_pairs.append(
                {
                    "question": current_question,
                    "answer": "\n".join(current_answer).strip(),
                    "metadata": {},
                }
            )

        return qa_pairs

    def _parse_numbered_qa(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse numbered question-answer pairs.

        Args:
            text: Numbered text

        Returns:
            List of question-answer dictionaries
        """
        qa_pairs = []
        lines = text.split("\n")

        current_question = None
        current_answer = []

        for line in lines:
            line = line.strip()

            # Check for numbered questions (1., 2., etc.)
            if line and line[0].isdigit() and "." in line[:3]:
                # Save previous pair if exists
                if current_question and current_answer:
                    qa_pairs.append(
                        {
                            "question": current_question,
                            "answer": "\n".join(current_answer).strip(),
                            "metadata": {},
                        }
                    )

                # Start new question
                current_question = (
                    line.split(".", 1)[1].strip() if "." in line else line
                )
                current_answer = []

            # Collect answer lines
            elif current_question and line and not line.startswith("---"):
                current_answer.append(line)

        # Add the last pair
        if current_question and current_answer:
            qa_pairs.append(
                {
                    "question": current_question,
                    "answer": "\n".join(current_answer).strip(),
                    "metadata": {},
                }
            )

        return qa_pairs

    def generate_temporal_qa_set(
        self, content: str, num_questions: Optional[int] = None, max_tokens: int = 4000
    ) -> List[Dict[str, Any]]:
        """
        Generate temporal question-answer pairs from content.

        Args:
            content: Input content
            num_questions: Number of questions to generate (None for unlimited)
            max_tokens: Maximum tokens for the response

        Returns:
            List of temporal question-answer dictionaries
        """
        system_prompt = """You are an expert at generating temporal reasoning questions from content. 
        Generate questions that test understanding of:
        - When events occurred
        - Duration of events or periods
        - Chronological order of events
        - Temporal relationships between events
        - Historical context and timelines
        
        Focus on questions that require temporal reasoning and understanding of time-based relationships."""

        # Handle unlimited questions case
        if num_questions is None:
            question_count_text = "multiple"
        else:
            question_count_text = str(num_questions)

        user_prompt = """Based on the following content, generate {question_count} temporal reasoning questions and their answers.

Content:
{content}

Please format your response as a JSON array of objects, where each object has:
- "question": A temporal reasoning question
- "answer": The correct answer
- "metadata": Additional information about the question type (e.g., "point_in_time", "duration", "sequence", "comparison")

Example format:
[
  {{
    "question": "When did the event occur?",
    "answer": "The event occurred in 1945.",
    "metadata": {{"type": "point_in_time"}}
  }}
]"""

        return self.generate_qa_set(
            content=content,
            system_prompt=system_prompt,
            user_prompt=user_prompt.format(question_count=question_count_text),
            num_questions=num_questions,
            max_tokens=max_tokens,
        )

    def generate_factual_qa_set(
        self, content: str, num_questions: Optional[int] = None, max_tokens: int = 4000
    ) -> List[Dict[str, Any]]:
        """
        Generate factual question-answer pairs from content.

        Args:
            content: Input content
            num_questions: Number of questions to generate (None for unlimited)
            max_tokens: Maximum tokens for the response

        Returns:
            List of factual question-answer dictionaries
        """
        system_prompt = """You are an expert at generating factual questions from content. 
        Generate questions that test understanding of:
        - Key facts and details
        - Important people, places, and events
        - Definitions and concepts
        - Relationships and connections
        - Specific information and data points
        
        Focus on questions that test factual knowledge and comprehension."""

        # Handle unlimited questions case
        if num_questions is None:
            question_count_text = "multiple"
        else:
            question_count_text = str(num_questions)

        user_prompt = """Based on the following content, generate {question_count} factual questions and their answers.

Content:
{content}

Please format your response as a JSON array of objects, where each object has:
- "question": A factual question
- "answer": The correct answer
- "metadata": Additional information about the question type (e.g., "factual", "definition", "relationship")

Example format:
[
  {{
    "question": "What is the main topic?",
    "answer": "The main topic is...",
    "metadata": {{"type": "factual"}}
  }}
]"""

        return self.generate_qa_set(
            content=content,
            system_prompt=system_prompt,
            user_prompt=user_prompt.format(question_count=question_count_text),
            num_questions=num_questions,
            max_tokens=max_tokens,
        )


def generate_qa_from_content(
    content: str,
    system_prompt: str,
    user_prompt: str,
    provider: Literal["anthropic", "openai", "huggingface"] = "anthropic",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    num_questions: Optional[int] = None,
    max_tokens: int = 4000,
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate question-answer pairs from content.

    Args:
        content: Input content
        system_prompt: System prompt for the AI
        user_prompt: User prompt template (should include {content} placeholder)
        provider: LLM provider ("anthropic", "openai", or "huggingface")
        api_key: API key for the specified provider (optional)
        model: Model name to use (optional)
        device: Device to run Hugging Face models on ("cpu" or "cuda")
        cache_dir: Cache directory for Hugging Face models (optional)
        num_questions: Number of questions to generate (None for unlimited)
        max_tokens: Maximum tokens for the response

    Returns:
        List of question-answer dictionaries
    """
    generator = AIQuestionGenerator(
        provider=provider,
        api_key=api_key,
        model=model,
        device=device,
        cache_dir=cache_dir,
    )
    return generator.generate_qa_set(
        content=content,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        num_questions=num_questions,
        max_tokens=max_tokens,
    )


def process_parquet_file(
    parquet_file_path: str,
    system_prompt: str,
    user_prompt: str,
    provider: Literal["anthropic", "openai", "huggingface"] = "anthropic",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    num_questions_per_content: Optional[int] = None,
    max_tokens: int = 4000,
    content_column: str = "content",
    success_column: str = "success",
    min_content_length: int = 100,
    use_batch: bool = False,
    url_column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Process a parquet file containing Wikipedia content through the AI question generator.

    Args:
        parquet_file_path: Path to the parquet file
        system_prompt: System prompt for the AI
        user_prompt: User prompt template (should include {content} placeholder)
        provider: LLM provider ("anthropic", "openai", or "huggingface")
        api_key: API key for the specified provider (optional)
        model: Model name to use (optional)
        device: Device to run Hugging Face models on ("cpu" or "cuda")
        cache_dir: Cache directory for Hugging Face models (optional)
        num_questions_per_content: Number of questions to generate per content item (None for unlimited)
        max_tokens: Maximum tokens for the response
        content_column: Name of the column containing the content to process
        success_column: Name of the column indicating successful extraction (optional)
        min_content_length: Minimum content length to process (filters out very short content)
        use_batch: Whether to use batch processing (only for Anthropic provider)
        url_column: Name of the column containing source URLs (used as custom_id for batching)

    Returns:
        List of all generated question-answer dictionaries from all content items
    """
    import pandas as pd

    # Initialize the AI question generator
    generator = AIQuestionGenerator(
        provider=provider,
        api_key=api_key,
        model=model,
        device=device,
        cache_dir=cache_dir,
    )

    # Read the parquet file
    try:
        df = pd.read_parquet(parquet_file_path)
        generator.log_info(f"Loaded parquet file with {len(df)} rows")
    except Exception as e:
        generator.log_error(f"Error reading parquet file: {e}")
        return []

    # Validate required columns
    if content_column not in df.columns:
        generator.log_error(
            f"Content column '{content_column}' not found in parquet file"
        )
        return []

    all_qa_pairs = []
    processed_count = 0
    skipped_count = 0

    # Check if we should use batch processing
    if use_batch and provider in ["anthropic", "openai"]:
        generator.log_info(f"Using {provider} batch processing")

        # Collect all valid content items for batch processing
        batch_contents = []
        row_data_map = {}  # Map to store original row data

        for idx, row in df.iterrows():
            # Check if content extraction was successful (if success column exists)
            if success_column in df.columns and not row.get(success_column, True):
                generator.log_info(
                    f"Skipping row {idx}: content extraction was not successful"
                )
                skipped_count += 1
                continue

            # Get content
            content = str(row[content_column]).strip()

            # Skip if content is too short
            if len(content) < min_content_length:
                generator.log_info(
                    f"Skipping row {idx}: content too short ({len(content)} chars)"
                )
                skipped_count += 1
                continue

            # Skip if content is empty or just whitespace
            if not content or content.isspace():
                generator.log_info(f"Skipping row {idx}: empty content")
                skipped_count += 1
                continue

            # Prepare content for batch processing
            content_item = {
                "content": content,
                "custom_id": (
                    str(row[url_column])
                    if url_column and url_column in df.columns
                    else f"row_{idx}"
                ),
            }

            batch_contents.append(content_item)
            row_data_map[content_item["custom_id"]] = {
                "row_index": idx,
                "row_data": row.to_dict(),
            }

        if batch_contents:
            try:
                # Create batch
                batch_info = generator.generate_qa_batch(
                    contents=batch_contents,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    num_questions=num_questions_per_content,
                )

                generator.log_info(f"Batch created with ID: {batch_info['batch_id']}")

                ## remove this, do not wait for the batch
                # Wait for batch completion
                # batch_results = generator.wait_for_batch_completion(batch_info['batch_id'])

                # # Process results and add metadata
                # for qa_pair in batch_results['qa_pairs']:
                #     custom_id = qa_pair.get('source_custom_id', '')
                #     if custom_id in row_data_map:
                #         row_info = row_data_map[custom_id]
                #         qa_pair['source_row_index'] = row_info['row_index']
                #         # Add all other columns from the original row as metadata
                #         for col in df.columns:
                #             if col != content_column:
                #                 qa_pair[f'source_{col}'] = row_info['row_data'][col]

                # all_qa_pairs = batch_results['qa_pairs']
                # processed_count = batch_results['successful_requests']
                # skipped_count = batch_results['failed_requests']

                # generator.log_info(f"Batch processing complete: {processed_count} successful, {skipped_count} failed")
                return []
            except Exception as e:
                generator.log_error(f"Error in batch processing: {e}")
                return []
        else:
            generator.log_info("No valid content found for batch processing")
    else:
        # Process individually (original logic)
        for idx, row in df.iterrows():
            # Check if content extraction was successful (if success column exists)
            if success_column in df.columns and not row.get(success_column, True):
                generator.log_info(
                    f"Skipping row {idx}: content extraction was not successful"
                )
                skipped_count += 1
                continue

            # Get content
            content = str(row[content_column]).strip()

            # Skip if content is too short
            if len(content) < min_content_length:
                generator.log_info(
                    f"Skipping row {idx}: content too short ({len(content)} chars)"
                )
                skipped_count += 1
                continue

            # Skip if content is empty or just whitespace
            if not content or content.isspace():
                generator.log_info(f"Skipping row {idx}: empty content")
                skipped_count += 1
                continue

            try:
                # Generate QA pairs for this content
                qa_pairs = generator.generate_qa_set(
                    content=content,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    num_questions=num_questions_per_content,
                    max_tokens=max_tokens,
                )

                # Add metadata from the original row
                for qa_pair in qa_pairs:
                    qa_pair["source_row_index"] = idx
                    # Add all other columns from the original row as metadata
                    for col in df.columns:
                        if col != content_column:
                            qa_pair[f"source_{col}"] = row[col]

                all_qa_pairs.extend(qa_pairs)
                processed_count += 1

                generator.log_info(
                    f"Processed row {idx}: generated {len(qa_pairs)} QA pairs"
                )

            except Exception as e:
                generator.log_error(f"Error processing row {idx}: {e}")
                skipped_count += 1
                continue

    generator.log_info(
        f"Processing complete: {processed_count} rows processed, {skipped_count} rows skipped, {len(all_qa_pairs)} total QA pairs generated"
    )
    return all_qa_pairs


def process_parquet_file_batch(
    parquet_file_path: str,
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    num_questions_per_content: Optional[int] = None,
    max_tokens: int = 4000,
    content_column: str = "content",
    success_column: str = "success",
    min_content_length: int = 100,
    url_column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to process a parquet file using Anthropic batch processing.

    Args:
        parquet_file_path: Path to the parquet file
        system_prompt: System prompt for the AI
        user_prompt: User prompt template (should include {content} placeholder)
        api_key: Anthropic API key (optional)
        model: Model name to use (optional)
        num_questions_per_content: Number of questions to generate per content item (None for unlimited)
        max_tokens: Maximum tokens for the response
        content_column: Name of the column containing the content to process
        success_column: Name of the column indicating successful extraction (optional)
        min_content_length: Minimum content length to process (filters out very short content)
        url_column: Name of the column containing source URLs (used as custom_id for batching)

    Returns:
        List of all generated question-answer dictionaries from all content items
    """
    return process_parquet_file(
        parquet_file_path=parquet_file_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider="anthropic",
        api_key=api_key,
        model=model,
        num_questions_per_content=num_questions_per_content,
        max_tokens=max_tokens,
        content_column=content_column,
        success_column=success_column,
        min_content_length=min_content_length,
        use_batch=True,
        url_column=url_column,
    )


def create_openai_batch_jsonl(
    contents: List[Dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 4000,
    output_file_path: str = "batch_requests.jsonl",
) -> str:
    """
    Create a JSONL file for OpenAI batch processing.

    This function creates a JSONL file that can be uploaded to OpenAI for batch processing.
    The file contains one JSON object per line, where each object represents a request.

    Args:
        contents: List of dictionaries containing 'content' and 'custom_id' keys
        system_prompt: System prompt for the AI
        user_prompt: User prompt template (should include {content} placeholder)
        model: OpenAI model to use
        max_tokens: Maximum tokens for the response
        output_file_path: Path where to save the JSONL file

    Returns:
        Path to the created JSONL file

    Example:
        contents = [
            {"content": "Some content here", "custom_id": "content_1"},
            {"content": "More content here", "custom_id": "content_2"}
        ]

        jsonl_path = create_openai_batch_jsonl(
            contents=contents,
            system_prompt="Generate questions from the content",
            user_prompt="Based on: {content}",
            output_file_path="my_batch.jsonl"
        )
    """
    import json

    with open(output_file_path, "w", encoding="utf-8") as f:
        for i, content_item in enumerate(contents):
            content = content_item["content"]
            custom_id = content_item.get("custom_id", f"content_{i}")

            # Format the user prompt with content
            formatted_user_prompt = user_prompt.format(content=content)

            # Create request according to OpenAI batch API specification
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "max_completion_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": formatted_user_prompt},
                    ],
                },
            }

            # Write the JSON object as a single line
            f.write(json.dumps(request) + "\n")

    return output_file_path


def create_openai_batch_from_parquet(
    parquet_file_path: str,
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 4000,
    content_column: str = "content",
    success_column: str = "success",
    min_content_length: int = 100,
    url_column: Optional[str] = None,
    output_file_path: str = "batch_requests.jsonl",
) -> str:
    """
    Create a JSONL file for OpenAI batch processing from a parquet file.

    Args:
        parquet_file_path: Path to the parquet file
        system_prompt: System prompt for the AI
        user_prompt: User prompt template (should include {content} placeholder)
        model: OpenAI model to use
        max_tokens: Maximum tokens for the response
        content_column: Name of the column containing the content to process
        success_column: Name of the column indicating successful extraction (optional)
        min_content_length: Minimum content length to process
        url_column: Name of the column containing source URLs (used as custom_id)
        output_file_path: Path where to save the JSONL file

    Returns:
        Path to the created JSONL file
    """
    import pandas as pd

    # Read the parquet file
    try:
        df = pd.read_parquet(parquet_file_path)
        print(f"Loaded parquet file with {len(df)} rows")
    except Exception as e:
        raise ValueError(f"Error reading parquet file: {e}")

    # Validate required columns
    if content_column not in df.columns:
        raise ValueError(f"Content column '{content_column}' not found in parquet file")

    # Collect valid content items
    contents = []
    skipped_count = 0

    for idx, row in df.iterrows():
        # Check if content extraction was successful (if success column exists)
        if success_column in df.columns and not row.get(success_column, True):
            print(f"Skipping row {idx}: content extraction was not successful")
            skipped_count += 1
            continue

        # Get content
        content = str(row[content_column]).strip()

        # Skip if content is too short
        if len(content) < min_content_length:
            print(f"Skipping row {idx}: content too short ({len(content)} chars)")
            skipped_count += 1
            continue

        # Skip if content is empty or just whitespace
        if not content or content.isspace():
            print(f"Skipping row {idx}: empty content")
            skipped_count += 1
            continue

        # Prepare content for batch processing
        content_item = {
            "content": content,
            "custom_id": (
                str(row[url_column])
                if url_column and url_column in df.columns
                else f"row_{idx}"
            ),
        }

        contents.append(content_item)

    print(
        f"Prepared {len(contents)} content items for batch processing (skipped {skipped_count})"
    )

    # Create the JSONL file
    return create_openai_batch_jsonl(
        contents=contents,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        max_tokens=max_tokens,
        output_file_path=output_file_path,
    )


if __name__ == "__main__":
    # Example usage
    import os

    # Example content
    content = """
    The Apollo 11 mission was the first manned mission to land on the Moon. 
    The mission launched on July 16, 1969, and landed on the Moon on July 20, 1969. 
    Neil Armstrong and Buzz Aldrin walked on the Moon while Michael Collins remained in orbit. 
    The mission lasted 8 days, 3 hours, 18 minutes, and 35 seconds.
    """

    # Example with Anthropic Claude
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        print("=== Using Anthropic Claude ===")
        generator_anthropic = AIQuestionGenerator(
            provider="anthropic", api_key=anthropic_api_key
        )

        # Generate temporal questions
        temporal_qa = generator_anthropic.generate_temporal_qa_set(
            content, num_questions=2
        )
        print("Temporal Questions (Anthropic):")
        for i, qa in enumerate(temporal_qa, 1):
            print(f"{i}. Q: {qa['question']}")
            print(f"   A: {qa['answer']}")
            print()
    else:
        print("ANTHROPIC_API_KEY not set, skipping Anthropic example")

    # Example with OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        print("=== Using OpenAI ===")
        generator_openai = AIQuestionGenerator(
            provider="openai", api_key=openai_api_key
        )

        # Generate factual questions
        factual_qa = generator_openai.generate_factual_qa_set(content, num_questions=2)
        print("Factual Questions (OpenAI):")
        for i, qa in enumerate(factual_qa, 1):
            print(f"{i}. Q: {qa['question']}")
            print(f"   A: {qa['answer']}")
            print()
    else:
        print("OPENAI_API_KEY not set, skipping OpenAI example")

    # Example with Hugging Face (default cache)
    print("=== Using Hugging Face (default cache) ===")
    try:
        generator_hf = AIQuestionGenerator(
            provider="huggingface", model="gpt2", device="cpu"
        )

        # Generate factual questions
        factual_qa = generator_hf.generate_factual_qa_set(content, num_questions=2)
        print("Factual Questions (Hugging Face - default cache):")
        for i, qa in enumerate(factual_qa, 1):
            print(f"{i}. Q: {qa['question']}")
            print(f"   A: {qa['answer']}")
            print()
    except Exception as e:
        print(f"Hugging Face example failed: {e}")

    # Example with Hugging Face (custom cache directory)
    print("=== Using Hugging Face (custom cache) ===")
    try:
        generator_hf_custom = AIQuestionGenerator(
            provider="huggingface",
            model="distilgpt2",
            device="cpu",
            cache_dir="./models",  # Custom cache directory
        )

        # Generate factual questions
        factual_qa = generator_hf_custom.generate_factual_qa_set(
            content, num_questions=2
        )
        print("Factual Questions (Hugging Face - custom cache):")
        for i, qa in enumerate(factual_qa, 1):
            print(f"{i}. Q: {qa['question']}")
            print(f"   A: {qa['answer']}")
            print()
    except Exception as e:
        print(f"Hugging Face custom cache example failed: {e}")

    if not anthropic_api_key and not openai_api_key:
        print("Note: No API keys set, but Hugging Face models can still be used")

    # Example of batch processing with parquet file
    print("\n=== Anthropic Batch Processing Example ===")
    if anthropic_api_key:
        print("Example batch processing command:")
        print("qa_pairs = process_parquet_file_batch(")
        print("    parquet_file_path='data/wikipedia_content.parquet',")
        print("    system_prompt='Generate temporal questions...',")
        print("    user_prompt='Based on: {content}',")
        print("    url_column='wikipedia_url'  # Used as custom_id")
        print(")")
        print("\nOr use the main function with batch processing:")
        print("qa_pairs = process_parquet_file(")
        print("    parquet_file_path='data/wikipedia_content.parquet',")
        print("    system_prompt='Generate temporal questions...',")
        print("    user_prompt='Based on: {content}',")
        print("    provider='anthropic',")
        print("    use_batch=True,")
        print("    url_column='wikipedia_url'")
        print(")")
    else:
        print("No Anthropic API key available for batch processing example")
