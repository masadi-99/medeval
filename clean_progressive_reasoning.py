#!/usr/bin/env python3
"""
Clean Progressive Reasoning System - FIXED VERSION
Based on user's exact specifications with critical fixes:

Step 0: History (inputs 1-4) + possible diagnoses list → Choose k=3 top candidates
Step 1: Exams/results (inputs 5-6) + k flowcharts → Choose starting diagnosis FROM FLOWCHART FIRST STEPS
Step 2-n: Patient info + flowchart position → Iterate through flowchart until leaf node

CRITICAL FIXES:
- Stage 3 chooses from flowchart FIRST STEPS (e.g., "Suspected Pneumonia") not categories
- Flowchart first steps include signs/symptoms/risks for proper LLM decision making
- Stage 4 properly iterates through flowchart until leaf node or max steps
- Single prompts_and_responses array (no redundant reasoning_trace)
"""

from typing import Dict, List, Optional, Tuple
import re
import json
import asyncio

# Import utility functions directly to avoid circular imports
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from medeval.utils import (
        load_flowchart_content,
        get_flowchart_knowledge, 
        get_flowchart_structure,
        get_flowchart_children,
        is_leaf_diagnosis,
        get_flowchart_first_step
    )
except ImportError:
    # Fallback: try direct import from local file
    from utils import (
        load_flowchart_content,
        get_flowchart_knowledge, 
        get_flowchart_structure,
        get_flowchart_children,
        is_leaf_diagnosis,
        get_flowchart_first_step
    )


class CleanProgressiveReasoning:
    """
    Clean implementation of progressive clinical reasoning - FIXED VERSION.
    Follows user's exact specifications step by step with critical fixes.
    """
    
    def __init__(self, evaluator):
        """Initialize with reference to main evaluator for LLM access"""
        self.evaluator = evaluator
        self.flowchart_categories = evaluator.flowchart_categories
        self.possible_diagnoses = evaluator.possible_diagnoses
        self.flowchart_dir = evaluator.flowchart_dir
    
    def run_progressive_workflow(self, sample: Dict, num_candidates: int = 3, 
                               max_reasoning_steps: int = 5) -> Dict:
        """
        Main entry point for FIXED clean progressive reasoning workflow.
        
        Args:
            sample: Clinical sample data
            num_candidates: Number of initial candidates (k=3)
            max_reasoning_steps: Max steps after step 1
            
        Returns:
            Complete results with all prompts/responses saved (NO REDUNDANCY)
        """
        
        # Track ALL steps with prompts and responses - SINGLE SOURCE OF TRUTH
        prompts_and_responses = []
        
        try:
            # STEP 0: History → Choose k candidate diagnoses
            print(f"🔍 Step 0: Analyzing history to choose {num_candidates} candidates...")
            step0_result = self._step0_choose_candidates(sample, num_candidates)
            prompts_and_responses.append(step0_result)
            
            if not step0_result.get('chosen_candidates'):
                return self._create_failure_result("Step 0 failed: No candidates chosen", prompts_and_responses)
            
            # STEP 1: Exams/results + flowcharts → Choose starting diagnosis FROM FLOWCHART FIRST STEPS
            print("🔍 Step 1: Analyzing complete clinical data with flowcharts...")
            step1_result = self._step1_choose_flowchart_first_step(sample, step0_result['chosen_candidates'])
            prompts_and_responses.append(step1_result)
            
            if not step1_result.get('chosen_first_step'):
                return self._create_failure_result("Step 1 failed: No first step chosen", prompts_and_responses)
            
            # STEP 2-n: Flowchart-guided reasoning to final diagnosis
            print(f"🔍 Steps 2-{max_reasoning_steps + 1}: Following flowchart to final diagnosis...")
            flowchart_steps = self._steps2_n_flowchart_iteration(
                sample, 
                step1_result['chosen_first_step'],
                step1_result['flowchart_category'], 
                max_reasoning_steps
            )
            prompts_and_responses.extend(flowchart_steps)
            
            # Determine final diagnosis from last step
            final_diagnosis = prompts_and_responses[-1].get('current_diagnosis', step1_result['chosen_first_step'])
            matched_diagnosis = self.evaluator.find_best_match(final_diagnosis)
            
            print(f"✅ Workflow complete: {len(prompts_and_responses)} steps, final diagnosis: {matched_diagnosis}")
            
            return {
                'final_diagnosis': matched_diagnosis,
                'reasoning_steps': len(prompts_and_responses),
                'suspicions': step0_result['chosen_candidates'],
                'recommended_tests': step1_result.get('flowcharts_requested', ''),
                'chosen_suspicion': step1_result['chosen_first_step'],  # FIXED: First step not category
                'reasoning_successful': bool(matched_diagnosis),
                'prompts_and_responses': prompts_and_responses,  # SINGLE SOURCE OF TRUTH
                'reasoning_trace': prompts_and_responses,  # For compatibility with main evaluator
                'mode': 'clean_step_by_step_fixed'
            }
            
        except Exception as e:
            print(f"❌ Error in progressive workflow: {e}")
            return self._create_failure_result(f"Workflow error: {e}", prompts_and_responses)
    
    async def run_progressive_workflow_async(self, sample: Dict, num_candidates: int = 3, 
                                           max_reasoning_steps: int = 5, request_prefix: str = "") -> Dict:
        """
        Async version of progressive reasoning workflow with proper request tracking.
        
        Args:
            sample: Clinical sample data
            num_candidates: Number of initial candidates (k=3)
            max_reasoning_steps: Max steps after step 1
            request_prefix: Unique prefix for tracking requests (e.g., sample path)
            
        Returns:
            Complete results with all prompts/responses saved (NO REDUNDANCY)
        """
        
        # Track ALL steps with prompts and responses - SINGLE SOURCE OF TRUTH
        prompts_and_responses = []
        
        try:
            # STEP 0: History → Choose k candidate diagnoses
            step0_result = await self._step0_choose_candidates_async(sample, num_candidates, f"{request_prefix}_step0")
            prompts_and_responses.append(step0_result)
            
            if not step0_result.get('chosen_candidates'):
                return self._create_failure_result("Step 0 failed: No candidates chosen", prompts_and_responses)
            
            # STEP 1: Exams/results + flowcharts → Choose starting diagnosis FROM FLOWCHART FIRST STEPS
            step1_result = await self._step1_choose_flowchart_first_step_async(
                sample, step0_result['chosen_candidates'], f"{request_prefix}_step1"
            )
            prompts_and_responses.append(step1_result)
            
            if not step1_result.get('chosen_first_step'):
                return self._create_failure_result("Step 1 failed: No first step chosen", prompts_and_responses)
            
            # STEP 2-n: Flowchart-guided reasoning to final diagnosis
            flowchart_steps = await self._steps2_n_flowchart_iteration_async(
                sample, 
                step1_result['chosen_first_step'],
                step1_result['flowchart_category'], 
                max_reasoning_steps,
                f"{request_prefix}_flowchart"
            )
            prompts_and_responses.extend(flowchart_steps)
            
            # Determine final diagnosis from last step
            final_diagnosis = prompts_and_responses[-1].get('current_diagnosis', step1_result['chosen_first_step'])
            matched_diagnosis = self.evaluator.find_best_match(final_diagnosis)
            
            return {
                'final_diagnosis': matched_diagnosis,
                'reasoning_steps': len(prompts_and_responses),
                'suspicions': step0_result['chosen_candidates'],
                'recommended_tests': step1_result.get('flowcharts_requested', ''),
                'chosen_suspicion': step1_result['chosen_first_step'],  # FIXED: First step not category
                'reasoning_successful': bool(matched_diagnosis),
                'prompts_and_responses': prompts_and_responses,  # SINGLE SOURCE OF TRUTH
                'reasoning_trace': prompts_and_responses,  # For compatibility with main evaluator
                'mode': 'clean_step_by_step_fixed_async'
            }
            
        except Exception as e:
            print(f"❌ Error in async progressive workflow: {e}")
            return self._create_failure_result(f"Async workflow error: {e}", prompts_and_responses)
    
    def _step0_choose_candidates(self, sample: Dict, num_candidates: int) -> Dict:
        """
        Step 0: History (inputs 1-4) + possible diagnoses list → Choose k top candidates
        """
        
        # Create history summary (inputs 1-4 only)
        history_summary = self._create_history_only_summary(sample)
        
        # Create Step 0 prompt
        prompt = self._create_step0_prompt(history_summary, num_candidates)
        
        # Query LLM
        response = self.evaluator.query_llm(prompt, max_tokens=800)
        
        # Parse candidates
        chosen_candidates = self._parse_step0_candidates(response, num_candidates)
        
        return {
            'step': 0,
            'stage': 'step0_candidate_selection',
            'action': 'choose_candidates',
            'prompt': prompt,
            'response': response,
            'chosen_candidates': chosen_candidates,
            'history_summary': history_summary,
            'reasoning': self._extract_step0_reasoning(response)
        }
    
    async def _step0_choose_candidates_async(self, sample: Dict, num_candidates: int, request_id: str) -> Dict:
        """
        Async version: Step 0: History (inputs 1-4) + possible diagnoses list → Choose k top candidates
        """
        
        # Create history summary (inputs 1-4 only)
        history_summary = self._create_history_only_summary(sample)
        
        # Create Step 0 prompt
        prompt = self._create_step0_prompt(history_summary, num_candidates)
        
        # Query LLM async with request tracking
        response_data = await self.evaluator.query_llm_async(prompt, request_id, max_tokens=800)
        
        if not response_data['success']:
            raise Exception(f"Step 0 API call failed: {response_data['error']}")
        
        response = response_data['response']
        
        # Parse candidates
        chosen_candidates = self._parse_step0_candidates(response, num_candidates)
        
        return {
            'step': 0,
            'stage': 'step0_candidate_selection',
            'action': 'choose_candidates',
            'prompt': prompt,
            'response': response,
            'chosen_candidates': chosen_candidates,
            'history_summary': history_summary,
            'reasoning': self._extract_step0_reasoning(response),
            'request_id': request_id
        }
    
    def _step1_choose_flowchart_first_step(self, sample: Dict, candidates: List[str]) -> Dict:
        """
        IMPROVED Step 1: Load flowcharts and ask LLM to choose from initial diagnoses
        """
        
        # Get complete clinical information (all 6 inputs)
        full_summary = self.evaluator.create_patient_data_summary(sample, 6)
        
        # Load complete flowcharts for candidates
        flowchart_info, loaded_flowcharts = self._load_complete_flowcharts(candidates)
        
        # Create Step 1 prompt with complete flowchart structures
        prompt = self._create_step1_improved_prompt(full_summary, flowchart_info)
        
        # Query LLM
        response = self.evaluator.query_llm(prompt, max_tokens=1000)
        
        # Parse chosen FIRST STEP and flowchart
        chosen_first_step, flowchart_category = self._parse_step1_first_step_choice(
            response, flowchart_info, loaded_flowcharts
        )
        
        return {
            'step': 1,
            'stage': 'step1_flowchart_first_step_selection',
            'action': 'choose_flowchart_first_step',
            'prompt': prompt,
            'response': response,
            'chosen_first_step': chosen_first_step,  # e.g., "Suspected Pneumonia"
            'flowchart_category': flowchart_category,  # e.g., "Pneumonia"
            'reasoning': self._extract_step1_reasoning(response)
        }
    
    async def _step1_choose_flowchart_first_step_async(self, sample: Dict, candidates: List[str], request_id: str) -> Dict:
        """
        Async version: IMPROVED Step 1: Load flowcharts and ask LLM to choose from initial diagnoses
        """
        
        # Get complete clinical information (all 6 inputs)
        full_summary = self.evaluator.create_patient_data_summary(sample, 6)
        
        # Load complete flowcharts for candidates
        flowchart_info, loaded_flowcharts = self._load_complete_flowcharts(candidates)
        
        # Create Step 1 prompt with complete flowchart structures
        prompt = self._create_step1_improved_prompt(full_summary, flowchart_info)
        
        # Query LLM async with request tracking
        response_data = await self.evaluator.query_llm_async(prompt, request_id, max_tokens=1000)
        
        if not response_data['success']:
            raise Exception(f"Step 1 API call failed: {response_data['error']}")
        
        response = response_data['response']
        
        # Parse chosen FIRST STEP and flowchart
        chosen_first_step, flowchart_category = self._parse_step1_first_step_choice(
            response, flowchart_info, loaded_flowcharts
        )
        
        return {
            'step': 1,
            'stage': 'step1_flowchart_first_step_selection',
            'action': 'choose_flowchart_first_step',
            'prompt': prompt,
            'response': response,
            'chosen_first_step': chosen_first_step,  # e.g., "Suspected Pneumonia"
            'flowchart_category': flowchart_category,  # e.g., "Pneumonia"
            'reasoning': self._extract_step1_reasoning(response),
            'request_id': request_id
        }
    
    def _steps2_n_flowchart_iteration(self, sample: Dict, starting_first_step: str, 
                                     flowchart_category: str, max_steps: int) -> List[Dict]:
        """
        FIXED Steps 2-n: Patient info + flowchart position → Iterate step by step until leaf node
        """
        
        steps = []
        current_node = starting_first_step  # Start from chosen first step
        step_number = 2
        
        # Get complete clinical information
        full_summary = self.evaluator.create_patient_data_summary(sample, 6)
        
        # Load flowchart structure
        try:
            flowchart_data = load_flowchart_content(flowchart_category, self.flowchart_dir)
            flowchart_structure = get_flowchart_structure(flowchart_data)
            flowchart_knowledge = get_flowchart_knowledge(flowchart_data)
        except Exception as e:
            print(f"Warning: Could not load flowchart for {flowchart_category}: {e}")
            return [{
                'step': 2,
                'stage': 'step2_flowchart_unavailable',
                'action': 'flowchart_unavailable',
                'response': f"Flowchart for {flowchart_category} unavailable",
                'current_diagnosis': starting_first_step
            }]
        
        # Iterate through flowchart until leaf node or max steps
        while step_number <= max_steps + 1:
            print(f"   📍 Step {step_number}: Current node: {current_node}")
            
            # Check if we've reached a leaf diagnosis
            if is_leaf_diagnosis(flowchart_structure, current_node):
                print(f"   🎯 Reached leaf diagnosis: {current_node}")
                break
            
            # Get next possible steps in flowchart
            next_options = get_flowchart_children(flowchart_structure, current_node)
            
            if not next_options:
                print(f"   🏁 No further options from: {current_node}")
                break
            
            # Reason through flowchart step
            step_result = self._reason_flowchart_step(
                sample, current_node, next_options, step_number,
                full_summary, flowchart_knowledge
            )
            steps.append(step_result)
            
            # Move to chosen next step
            current_node = step_result.get('chosen_diagnosis', current_node)
            step_number += 1
        
        return steps
    
    async def _steps2_n_flowchart_iteration_async(self, sample: Dict, starting_first_step: str, 
                                                 flowchart_category: str, max_steps: int, request_prefix: str) -> List[Dict]:
        """
        Async version: FIXED Steps 2-n: Patient info + flowchart position → Iterate step by step until leaf node
        """
        
        steps = []
        current_node = starting_first_step  # Start from chosen first step
        step_number = 2
        
        # Get complete clinical information
        full_summary = self.evaluator.create_patient_data_summary(sample, 6)
        
        # Load flowchart structure
        try:
            flowchart_data = load_flowchart_content(flowchart_category, self.flowchart_dir)
            flowchart_structure = get_flowchart_structure(flowchart_data)
            flowchart_knowledge = get_flowchart_knowledge(flowchart_data)
        except Exception as e:
            print(f"Warning: Could not load flowchart for {flowchart_category}: {e}")
            return [{
                'step': 2,
                'stage': 'step2_flowchart_unavailable',
                'action': 'flowchart_unavailable',
                'response': f"Flowchart for {flowchart_category} unavailable",
                'current_diagnosis': starting_first_step
            }]
        
        # Iterate through flowchart until leaf node or max steps
        while step_number <= max_steps + 1:
            # Check if we've reached a leaf diagnosis
            if is_leaf_diagnosis(flowchart_structure, current_node):
                break
            
            # Get next possible steps in flowchart
            next_options = get_flowchart_children(flowchart_structure, current_node)
            
            if not next_options:
                break
            
            # Reason through flowchart step async
            step_result = await self._reason_flowchart_step_async(
                sample, current_node, next_options, step_number,
                full_summary, flowchart_knowledge, f"{request_prefix}_step{step_number}"
            )
            steps.append(step_result)
            
            # Move to chosen next step
            current_node = step_result.get('chosen_diagnosis', current_node)
            step_number += 1
        
        return steps
    
    def _reason_flowchart_step(self, sample: Dict, current_node: str,
                              next_options: List[str], step_number: int,
                              full_summary: str, flowchart_knowledge: Dict) -> Dict:
        """
        Reason through a single flowchart step with evidence matching
        """
        
        prompt = self._create_flowchart_step_prompt(
            full_summary, current_node, next_options, step_number, flowchart_knowledge
        )
        
        response = self.evaluator.query_llm(prompt, max_tokens=800)
        
        chosen_diagnosis = self._parse_flowchart_step_choice(response, next_options)
        
        return {
            'step': step_number,
            'stage': f'step{step_number}_flowchart_reasoning',
            'action': 'flowchart_step_reasoning',
            'prompt': prompt,
            'response': response,
            'current_node': current_node,
            'next_options': next_options,
            'chosen_diagnosis': chosen_diagnosis,
            'current_diagnosis': chosen_diagnosis,  # For final diagnosis extraction
            'reasoning': self._extract_flowchart_reasoning(response)
        }
    
    async def _reason_flowchart_step_async(self, sample: Dict, current_node: str,
                                         next_options: List[str], step_number: int,
                                         full_summary: str, flowchart_knowledge: Dict, request_id: str) -> Dict:
        """
        Async version: Reason through one step of flowchart navigation
        """
        
        # Create flowchart step prompt
        prompt = self._create_flowchart_step_prompt(
            full_summary, current_node, next_options, step_number, flowchart_knowledge
        )
        
        # Query LLM async with request tracking
        response_data = await self.evaluator.query_llm_async(prompt, request_id, max_tokens=800)
        
        if not response_data['success']:
            # Fallback to first option if API call fails
            chosen_diagnosis = next_options[0] if next_options else current_node
            response = f"API call failed, defaulting to: {chosen_diagnosis}"
        else:
            response = response_data['response']
            chosen_diagnosis = self._parse_flowchart_step_choice(response, next_options)
        
        return {
            'step': step_number,
            'stage': f'step{step_number}_flowchart_reasoning',
            'action': 'flowchart_step_reasoning',
            'prompt': prompt,
            'response': response,
            'current_node': current_node,
            'next_options': next_options,
            'chosen_diagnosis': chosen_diagnosis,
            'current_diagnosis': chosen_diagnosis,
            'reasoning': self._extract_flowchart_reasoning(response),
            'request_id': request_id
        }
    
    # === PROMPT CREATION METHODS ===
    
    def _create_step0_prompt(self, history_summary: str, num_candidates: int) -> str:
        """Create Step 0 prompt for choosing candidates based on history only"""
        
        prompt = f"""You are a medical expert analyzing patient history to identify initial diagnostic candidates.

**Patient History (Initial Information Only):**
{history_summary}

**Available Disease Categories (EXACT NAMES TO CHOOSE FROM):**
{chr(10).join(f'{i+1}. {cat}' for i, cat in enumerate(self.flowchart_categories))}

**Task:** Based on the patient history above, select the {num_candidates} most likely disease categories.

**CRITICAL INSTRUCTIONS:**
• You MUST choose EXACTLY from the numbered list above - use the EXACT names provided
• Do NOT use synonyms, abbreviations, or full names if they differ from the list
• Do NOT invent new category names or use alternative spellings
• Focus only on the patient history provided (no additional test results yet)
• Choose categories that most likely match the presenting symptoms and history
• Provide reasoning for each choice based on historical clinical findings
• Rank them in order of likelihood

**Examples of CORRECT responses:**
• "COPD" (if "COPD" is in the list) - NOT "Chronic Obstructive Pulmonary Disease"
• "Acute Coronary Syndrome" (if listed) - NOT "ACS" or "Heart Attack"
• "Pulmonary Embolism" (if listed) - NOT "PE" or "Blood Clot"

**Format:**
**CLINICAL REASONING:**
[Systematic analysis of history against potential categories]

**FINAL CANDIDATES:**
1. [EXACT category name from numbered list above with brief justification]
2. [EXACT category name from numbered list above with brief justification]  
3. [EXACT category name from numbered list above with brief justification]

**DETAILED REASONING:** [Complete explanation of choice rationale using only the exact category names from the list]"""
        
        return prompt
    
    def _create_step1_improved_prompt(self, full_summary: str, flowchart_info: str) -> str:
        """Create Step 1 prompt for choosing from flowchart diagnostic structures with clinical knowledge"""
        
        prompt = f"""You are a medical expert with complete clinical information and access to diagnostic flowcharts with clinical knowledge.

**Complete Patient Clinical Information:**
{full_summary}

{flowchart_info}

**Task:** Choose the most likely initial diagnosis from the flowchart diagnostic structures above using the clinical knowledge provided.

**CRITICAL INSTRUCTIONS:**
• You must choose from the FIRST-LEVEL diagnoses in the diagnostic structures (e.g., "Suspected Pneumonia", "Suspected Pulmonary Embolism")
• These first-level diagnoses are the entry points to the flowcharts
• USE THE CLINICAL KNOWLEDGE provided for each suspected condition to guide your decision
• Match patient's clinical findings (symptoms, signs, risk factors) against the knowledge provided for each suspected condition
• Choose the initial diagnosis where the patient's presentation most likely matches the clinical knowledge criteria
• Focus on evidence-based matching between patient findings and the risk factors, symptoms, and signs listed

**Analysis Required:**
• Compare patient's clinical findings against the clinical knowledge (risk factors, symptoms, signs) for each suspected condition
• Identify which suspected diagnosis has the most likely clinical knowledge match with the patient presentation
• Use the provided symptoms, signs, and risk factors to guide your evidence matching
• Provide detailed evidence matching using the clinical knowledge criteria
• Explain why your chosen suspected diagnosis has superior clinical knowledge alignment

**Format:**
**CLINICAL KNOWLEDGE MATCHING:**
[Systematic comparison of patient findings against the risk factors, symptoms, and signs provided for each suspected condition]

**EVIDENCE-BASED REASONING:**
[Compare how well patient matches the clinical knowledge criteria for each suspected diagnosis and explain your choice]

**CHOSEN INITIAL DIAGNOSIS:** [Exact first-level diagnosis name from the diagnostic structures above - must start with "Suspected"]
**DETAILED REASONING:** [Complete medical justification using the clinical knowledge criteria (risk factors, symptoms, signs) provided above]"""
        
        return prompt
    
    def _create_flowchart_step_prompt(self, full_summary: str, current_node: str,
                                     next_options: List[str], step_number: int,
                                     flowchart_knowledge: Dict) -> str:
        """Create prompt for reasoning through flowchart step - IMPROVED to focus only on flowchart navigation"""
        
        prompt = f"""You are a medical expert following a diagnostic flowchart step-by-step to reach a final diagnosis.

**Complete Patient Clinical Information:**
{full_summary}

**Current Position in Flowchart:** {current_node}

**Available Next Steps in Flowchart:**"""
        
        for i, option in enumerate(next_options, 1):
            prompt += f"\n{i}. {option}"
        
        # Add flowchart knowledge for current node and options
        if flowchart_knowledge:
            prompt += f"\n\n**Relevant Clinical Knowledge from Flowchart:**"
            
            # Add knowledge for current node
            if current_node in flowchart_knowledge:
                knowledge_text = flowchart_knowledge[current_node]
                prompt += f"\n\n**Current Node ({current_node}):**"
                if isinstance(knowledge_text, dict):
                    for key, value in knowledge_text.items():
                        prompt += f"\n• {key}: {value}"
                else:
                    prompt += f"\n• {knowledge_text}"
            
            # Add knowledge for each next option
            for option in next_options:
                if option in flowchart_knowledge:
                    knowledge_text = flowchart_knowledge[option]
                    prompt += f"\n\n**{option}:**"
                    if isinstance(knowledge_text, dict):
                        for key, value in knowledge_text.items():
                            prompt += f"\n• {key}: {value}"
                    else:
                        prompt += f"\n• {knowledge_text}"
        
        prompt += f"""

**Task:** Choose the most likely next step in the flowchart based on patient's clinical findings.

**CRITICAL INSTRUCTIONS:**
• Follow the flowchart structure step-by-step until reaching a leaf node (final diagnosis)
• Base your decision ONLY on patient findings matching the clinical criteria
• Do NOT consider diagnoses outside this flowchart
• Continue the systematic flowchart progression
• Provide evidence-based reasoning for your choice

**Format:**
**EVIDENCE MATCHING:**"""
        
        for i, option in enumerate(next_options, 1):
            prompt += f"\n{i}. {option}: [How patient findings match/don't match clinical criteria]"
        
        prompt += f"""

**FLOWCHART REASONING:**
[Why chosen option most likely fits the flowchart progression and patient evidence]

**CHOSEN STEP:** [Number] - [Step name]
**REASONING:** [Complete medical justification based on flowchart criteria]"""
        
        return prompt
    
    def _create_single_step_direct_reasoning_prompt(self, full_summary: str) -> str:
        """Create prompt for single-step direct reasoning with all patient info and all possible diagnoses"""
        
        prompt = f"""You are a medical expert tasked with providing a primary discharge diagnosis based on complete clinical information.

**Complete Patient Clinical Information:**
{full_summary}

**All Possible Primary Discharge Diagnoses:**
{chr(10).join(f'{i+1}. {diagnosis}' for i, diagnosis in enumerate(self.possible_diagnoses))}

**Task:** Based on the complete clinical information provided above, determine the most likely primary discharge diagnosis from the possible diagnoses list.

**CRITICAL INSTRUCTIONS:**
• Analyze ALL the clinical information systematically (history, examination, laboratory/imaging results)
• Consider differential diagnoses and rule out alternatives
• Use evidence-based medical reasoning to support your choice
• You MUST choose from the numbered list of possible diagnoses above
• Provide detailed reasoning FIRST, then your final diagnosis
• Use the exact diagnosis name from the numbered list

**Required Analysis Structure:**

**SYSTEMATIC CLINICAL ANALYSIS:**
[Analyze the patient presentation systematically - demographics, chief complaint, history, physical findings, laboratory/imaging results]

**DIFFERENTIAL DIAGNOSIS CONSIDERATION:**
[Consider the main differential diagnoses based on the clinical presentation and explain why you are considering or ruling out each major possibility]

**EVIDENCE MATCHING:**
[Match the patient's specific clinical findings against the diagnostic criteria for your top differential diagnoses]

**SUPPORTING EVIDENCE:**
[List the specific clinical findings that strongly support your chosen diagnosis]

**CONTRADICTORY EVIDENCE:**
[Address any findings that might argue against your chosen diagnosis and explain why your diagnosis is still most likely]

**FINAL DIAGNOSIS:** [Exact diagnosis name from the numbered list above]

**DIAGNOSTIC REASONING:** [Complete medical justification for your final diagnosis choice, including why this diagnosis is more likely than the main alternatives]"""
        
        return prompt
    
    # === PARSING METHODS ===
    
    def _parse_step0_candidates(self, response: str, num_candidates: int) -> List[str]:
        """Parse chosen candidates from Step 0 response - prioritize exact matches"""
        
        candidates = []
        
        # Try to extract from FINAL CANDIDATES section
        final_match = re.search(r'FINAL CANDIDATES:\s*(.*?)(?=\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
        if final_match:
            lines = final_match.group(1).strip().split('\n')
            for line in lines:
                # Look for exact matches from flowchart categories first
                exact_candidate = self._extract_exact_category_match(line)
                if exact_candidate and exact_candidate not in candidates:
                    candidates.append(exact_candidate)
        
        # Fallback: numbered lists anywhere in response
        if len(candidates) < num_candidates:
            all_matches = re.findall(r'^\d+\.\s*(.+)', response, re.MULTILINE)
            for match in all_matches:
                exact_candidate = self._extract_exact_category_match(match)
                if exact_candidate and exact_candidate not in candidates:
                    candidates.append(exact_candidate)
                    if len(candidates) >= num_candidates:
                        break
        
        # Final fallback: use old parsing if needed
        if len(candidates) < num_candidates:
            all_matches = re.findall(r'^\d+\.\s*(.+)', response, re.MULTILINE)
            for match in all_matches:
                clean_candidate = self._extract_clean_category_name(match)
                if clean_candidate and clean_candidate not in candidates:
                    candidates.append(clean_candidate)
                    if len(candidates) >= num_candidates:
                        break
        
        # Ensure correct number
        while len(candidates) < num_candidates:
            candidates.append(f"Unknown Category {len(candidates) + 1}")
        
        return candidates[:num_candidates]
    
    def _extract_exact_category_match(self, text: str) -> str:
        """Extract exact category name from flowchart categories list"""
        
        # Remove markdown formatting and clean text
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        clean_text = clean_text.strip()
        
        # Try exact substring matching against flowchart categories
        for category in self.flowchart_categories:
            if category in clean_text:
                return category
        
        # Try case-insensitive matching
        for category in self.flowchart_categories:
            if category.lower() in clean_text.lower():
                return category
        
        return ""
    
    def _extract_clean_category_name(self, text: str) -> str:
        """Extract clean category name from LLM response text"""
        
        # Remove markdown formatting
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        clean_text = clean_text.strip()
        
        # First try to extract from numbered lists (most reliable)
        numbered_match = re.search(r'^\d+\.\s*([A-Za-z][A-Za-z\s\-]+?)(?:\s*-|\s*\(|$)', clean_text)
        if numbered_match:
            candidate = numbered_match.group(1).strip()
            if self._is_valid_medical_category(candidate):
                return self._map_to_flowchart_category(candidate)
        
        # Try to extract medical condition names (more specific patterns)
        medical_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Disease|\s+Syndrome|\s+Disorder|\s+Insufficiency|\s+Failure|\s+Embolism|\s+Dissection))\b',  # Formal medical names
            r'\b(COPD|ACS|PE|MS|GERD|AFib)\b',  # Common abbreviations
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Cancer|\s+Tumor|\s+Carcinoma))\b',  # Cancer terms
            r'\b(Pneumonia|Asthma|Migraine|Stroke|Diabetes|Hypertension|Tuberculosis|Epilepsy)\b',  # Common single-word conditions
        ]
        
        for pattern in medical_patterns:
            match = re.search(pattern, clean_text)
            if match:
                candidate = match.group(1).strip()
                if self._is_valid_medical_category(candidate):
                    mapped = self._map_to_flowchart_category(candidate)
                    # CRITICAL: Only return if it maps to a valid flowchart category
                    if mapped and mapped in self.flowchart_categories:
                        return mapped
        
        # Fallback: try original patterns but with better validation
        fallback_patterns = [
            r'(\d+\.\s*)?([A-Za-z][A-Za-z\s\-]+?)(?:\s*-|\s*\(|\Z)',  # Extract before dash or parenthesis
        ]
        
        for pattern in fallback_patterns:
            match = re.search(pattern, clean_text)
            if match:
                candidate = match.group(1) if match.group(1) and not match.group(1).strip().endswith('.') else match.group(2)
                candidate = candidate.strip()
                
                # Only accept if it looks like a medical condition AND maps to valid flowchart
                if self._is_valid_medical_category(candidate):
                    mapped = self._map_to_flowchart_category(candidate)
                    # CRITICAL: Only return if it maps to a valid flowchart category
                    if mapped and mapped in self.flowchart_categories:
                        return mapped
        
        return ""
    
    def _is_valid_medical_category(self, candidate: str) -> bool:
        """Check if a candidate string looks like a valid medical category"""
        
        # Too short or too long
        if len(candidate) < 3 or len(candidate) > 50:
            return False
        
        # Contains invalid phrases that suggest it's not a medical condition
        invalid_phrases = [
            'the nature of', 'are concerning for', 'suggests that', 'indicates that',
            'patient has', 'history of', 'symptoms of', 'signs of', 'evidence of',
            'likely to be', 'consistent with', 'compatible with', 'rule out'
        ]
        
        candidate_lower = candidate.lower()
        for phrase in invalid_phrases:
            if phrase in candidate_lower:
                return False
        
        # Must start with a capital letter or be a known abbreviation
        if not (candidate[0].isupper() or candidate.isupper()):
            return False
        
        # Should not contain certain words that suggest it's not a condition name
        invalid_words = ['her', 'his', 'the', 'this', 'that', 'which', 'when', 'where', 'how', 'why']
        words = candidate.lower().split()
        if any(word in invalid_words for word in words):
            return False
        
        return True
    
    def _map_to_flowchart_category(self, candidate: str) -> str:
        """Map a candidate to an actual flowchart category"""
        
        # Map common variations to actual flowchart categories
        category_mapping = {
            'Acute Coronary Syndrome': 'Acute Coronary Syndrome',
            'ACS': 'Acute Coronary Syndrome',
            'Aortic Dissection': 'Aortic Dissection', 
            'Pulmonary Embolism': 'Pulmonary Embolism',
            'PE': 'Pulmonary Embolism',
            'Heart Failure': 'Heart Failure',
            'Pneumonia': 'Pneumonia',
            'COPD': 'COPD',
            'Chronic Obstructive Pulmonary Disease': 'COPD',  # Map full name to abbreviation
            'Asthma': 'Asthma',
            'Gastroesophageal Reflux Disease': 'Gastro-oesophageal Reflux Disease',  # Handle spelling variations
            'GERD': 'Gastro-oesophageal Reflux Disease',
            'Multiple Sclerosis': 'Multiple Sclerosis',
            'MS': 'Multiple Sclerosis',
            'Atrial Fibrillation': 'Atrial Fibrillation',
            'AFib': 'Atrial Fibrillation',
            'A-fib': 'Atrial Fibrillation',
            'Chronic Pancreatitis': 'Peptic Ulcer Disease',  # Map to closest gastrointestinal category
            'Pancreatitis': 'Peptic Ulcer Disease',
            'Peptic Ulcer': 'Peptic Ulcer Disease',
            'Gastric Ulcer': 'Peptic Ulcer Disease',
            'Duodenal Ulcer': 'Peptic Ulcer Disease'
        }
        
        # Try exact match first
        if candidate in category_mapping:
            return category_mapping[candidate]
        
        # Try partial matching
        for key, value in category_mapping.items():
            if key.lower() in candidate.lower():
                return value
        
        # Check if it matches any of our available categories
        for category in self.flowchart_categories:
            if category.lower() in candidate.lower() or candidate.lower() in category.lower():
                return category
        
        # Return cleaned candidate if no mapping found
        if len(candidate) > 3:
            return candidate
        
        return ""
    
    def _parse_step1_first_step_choice(self, response: str, flowchart_info: str,
                                      loaded_flowcharts: Dict) -> Tuple[str, str]:
        """Parse chosen FIRST STEP from Step 1 response using diagnostic structures"""
        
        # Extract chosen initial diagnosis
        initial_diagnosis_match = re.search(r'CHOSEN INITIAL DIAGNOSIS:\s*(.+?)(?=\n|\Z)', response, re.IGNORECASE)
        
        # Fallback to old format
        if not initial_diagnosis_match:
            initial_diagnosis_match = re.search(r'CHOSEN STARTING POINT:\s*(.+?)(?=\n|\Z)', response, re.IGNORECASE)
        
        chosen_first_step = None
        flowchart_category = None
        
        if initial_diagnosis_match:
            chosen_text = initial_diagnosis_match.group(1).strip()
            
            # Find matching first step from loaded flowcharts using diagnostic structures
            for category, flowchart_data in loaded_flowcharts.items():
                try:
                    # Extract first-level keys from diagnostic structure
                    if 'diagnostic' in flowchart_data:
                        diagnostic_structure = flowchart_data['diagnostic']
                        # Get the first-level keys (like "Suspected Pneumonia")
                        for first_level_key in diagnostic_structure.keys():
                            if first_level_key.lower() in chosen_text.lower() or chosen_text.lower() in first_level_key.lower():
                                chosen_first_step = first_level_key
                                flowchart_category = category
                                break
                        if chosen_first_step:
                            break
                except:
                    continue
        
        # Fallbacks
        if not chosen_first_step:
            # Use first available flowchart's first-level key
            for category, flowchart_data in loaded_flowcharts.items():
                try:
                    if 'diagnostic' in flowchart_data:
                        diagnostic_structure = flowchart_data['diagnostic']
                        # Get the first key from the diagnostic structure
                        if diagnostic_structure:
                            chosen_first_step = list(diagnostic_structure.keys())[0]
                            flowchart_category = category
                            break
                except:
                    continue
        
        if not chosen_first_step:
            chosen_first_step = "Suspected Unknown Condition"
            flowchart_category = "Unknown"
        
        return chosen_first_step, flowchart_category
    
    def _parse_flowchart_step_choice(self, response: str, options: List[str]) -> str:
        """Parse chosen step from flowchart step response"""
        
        # First try flexible pattern for "CHOSEN STEP" with number (this works reliably)
        flexible_match = re.search(r'CHOSEN STEP.*?(\d+)', response, re.IGNORECASE)
        if flexible_match:
            try:
                choice_num = int(flexible_match.group(1))
                if 1 <= choice_num <= len(options):
                    return options[choice_num - 1]
            except ValueError:
                pass
        
        # Extract from CHOSEN STEP section (handle both with and without asterisks)
        chosen_match = re.search(r'\*?\*?CHOSEN STEP\*?\*?:\s*(\d+)\s*-\s*(.+?)(?=\n|\Z)', 
                               response, re.IGNORECASE)
        
        if chosen_match:
            try:
                choice_num = int(chosen_match.group(1))
                if 1 <= choice_num <= len(options):
                    return options[choice_num - 1]
            except ValueError:
                pass
        
        # Fallback: any number in the response
        number_match = re.search(r'\b(\d+)\b', response)
        if number_match:
            try:
                choice_num = int(number_match.group(1))
                if 1 <= choice_num <= len(options):
                    return options[choice_num - 1]
            except ValueError:
                pass
        
        return options[0] if options else "Unknown"
    
    def _parse_single_step_final_diagnosis(self, response: str) -> str:
        """Parse final diagnosis from single-step direct reasoning response"""
        
        # First, try to extract from FINAL DIAGNOSIS section
        final_diagnosis_match = re.search(r'FINAL DIAGNOSIS:\s*(.+?)(?=\n|\Z)', response, re.IGNORECASE)
        if final_diagnosis_match:
            diagnosis_text = final_diagnosis_match.group(1).strip()
            
            # Try to extract diagnosis name (remove any numbering if present)
            # Look for numbered format like "1. Diagnosis Name" or just "Diagnosis Name"
            clean_match = re.search(r'(?:\d+\.\s*)?(.+)', diagnosis_text)
            if clean_match:
                candidate_diagnosis = clean_match.group(1).strip()
                
                # Check if this matches any of our possible diagnoses
                matched = self._find_exact_diagnosis_match(candidate_diagnosis)
                if matched:
                    return matched
        
        # Fallback: look for numbered diagnosis in the response
        numbered_matches = re.findall(r'(\d+)\.\s*([^.\n]+)', response)
        for num_str, diagnosis_text in numbered_matches:
            try:
                num = int(num_str)
                if 1 <= num <= len(self.possible_diagnoses):
                    return self.possible_diagnoses[num - 1]
            except ValueError:
                continue
        
        # Final fallback: try to find any possible diagnosis mentioned in the text
        for diagnosis in self.possible_diagnoses:
            if diagnosis.lower() in response.lower():
                return diagnosis
        
        return "Unknown Diagnosis"
    
    def _find_exact_diagnosis_match(self, candidate: str) -> str:
        """Find exact match from possible diagnoses list"""
        
        # Remove common prefixes/suffixes and clean the text
        cleaned_candidate = re.sub(r'^(?:primary|discharge|final)?\s*(?:diagnosis:?)?\s*', '', candidate, flags=re.IGNORECASE)
        cleaned_candidate = cleaned_candidate.strip()
        
        # Try exact match first
        for diagnosis in self.possible_diagnoses:
            if diagnosis.lower() == cleaned_candidate.lower():
                return diagnosis
        
        # Try partial match
        for diagnosis in self.possible_diagnoses:
            if diagnosis.lower() in cleaned_candidate.lower() or cleaned_candidate.lower() in diagnosis.lower():
                return diagnosis
        
        return ""
    
    def _extract_single_step_reasoning(self, response: str) -> str:
        """Extract reasoning from single-step direct reasoning response"""
        
        # Try to extract everything before FINAL DIAGNOSIS
        final_diagnosis_pos = response.lower().find('final diagnosis:')
        if final_diagnosis_pos > 0:
            reasoning_text = response[:final_diagnosis_pos].strip()
        else:
            # Use the full response as reasoning
            reasoning_text = response.strip()
        
        # Clean up the reasoning text
        reasoning_text = re.sub(r'\*\*[^*]+\*\*\s*', '', reasoning_text)  # Remove section headers
        reasoning_text = ' '.join(reasoning_text.split())  # Normalize whitespace
        
        return reasoning_text if reasoning_text else response.strip()
    
    # === UTILITY METHODS ===
    
    def _create_history_only_summary(self, sample: Dict) -> str:
        """Create summary from first 4 inputs only (history)"""
        
        input_descriptions = {
            1: "Chief Complaint",
            2: "History of Present Illness",
            3: "Past Medical History",
            4: "Family History"
        }
        
        summary = ""
        for i in range(1, 5):  # Only inputs 1-4
            input_key = f"input{i}"
            if input_key in sample:
                content = sample[input_key]
                summary += f"• {input_descriptions[i]}: {content}\n"
        
        return summary.strip()
    
    def _load_complete_flowcharts(self, candidates: List[str]) -> Tuple[str, Dict]:
        """Load complete flowcharts for candidates and present raw diagnostic structures + clinical knowledge for selection"""
        
        flowchart_info = "**AVAILABLE DIAGNOSTIC FLOWCHARTS:**\n\n"
        loaded_flowcharts = {}
        
        for candidate in candidates:
            try:
                flowchart_data = load_flowchart_content(candidate, self.flowchart_dir)
                loaded_flowcharts[candidate] = flowchart_data
                
                # Extract and present the raw diagnostic structure
                if 'diagnostic' in flowchart_data:
                    diagnostic_structure = flowchart_data['diagnostic']
                    flowchart_info += f"**{candidate} Flowchart:**\n"
                    flowchart_info += f'"diagnostic": {diagnostic_structure}\n'
                    
                    # ENHANCEMENT: Also include clinical knowledge for first-level diagnoses
                    if 'knowledge' in flowchart_data:
                        knowledge_data = flowchart_data['knowledge']
                        # Extract knowledge for each first-level diagnostic key
                        for first_level_key in diagnostic_structure.keys():
                            if first_level_key in knowledge_data:
                                knowledge_content = knowledge_data[first_level_key]
                                if isinstance(knowledge_content, dict):
                                    flowchart_info += f"\n**{first_level_key} Clinical Knowledge:**\n"
                                    # Add risk factors, symptoms, signs if available
                                    for knowledge_type in ['Risk Factors', 'Symptoms', 'Signs']:
                                        if knowledge_type in knowledge_content:
                                            flowchart_info += f"• {knowledge_type}: {knowledge_content[knowledge_type]}\n"
                    
                    flowchart_info += "\n"
                else:
                    # Fallback if diagnostic structure not found
                    flowchart_info += f"**{candidate} Flowchart:**\n"
                    flowchart_info += f'"diagnostic": {{"Suspected {candidate}": {{"Unknown": []}}}}\n\n'
                    loaded_flowcharts[candidate] = {"diagnostic": {f"Suspected {candidate}": {"Unknown": []}}}
                
            except Exception as e:
                print(f"Warning: Could not load flowchart for {candidate}: {e}")
                flowchart_info += f"**{candidate} Flowchart:**\n"
                flowchart_info += f'"diagnostic": {{"Suspected {candidate}": {{"Unknown": []}}}}\n\n'
                loaded_flowcharts[candidate] = {"diagnostic": {f"Suspected {candidate}": {"Unknown": []}}}
        
        return flowchart_info.strip(), loaded_flowcharts
    
    def _extract_step0_reasoning(self, response: str) -> str:
        """Extract reasoning from Step 0 response"""
        match = re.search(r'DETAILED REASONING:\s*(.+?)(?=\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else response.strip()
    
    def _extract_step1_reasoning(self, response: str) -> str:
        """Extract reasoning from Step 1 response"""
        match = re.search(r'DETAILED REASONING:\s*(.+?)(?=\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else response.strip()
    
    def _extract_flowchart_reasoning(self, response: str) -> str:
        """Extract reasoning from flowchart step response"""
        match = re.search(r'REASONING:\s*(.+?)(?=\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else response.strip()
    
    def _create_failure_result(self, error_message: str, steps: List[Dict]) -> Dict:
        """Create failure result with error message"""
        return {
            'final_diagnosis': 'Error',
            'reasoning_steps': len(steps),
            'suspicions': [],
            'recommended_tests': '',
            'chosen_suspicion': error_message,
            'reasoning_successful': False,
            'prompts_and_responses': steps,
            'mode': 'clean_step_by_step_fixed_error'
        }

    def run_single_step_direct_reasoning(self, sample: Dict) -> Dict:
        """
        Single-step direct reasoning: All patient info + all possible diagnoses → Final diagnosis with reasoning.
        
        Args:
            sample: Clinical sample data
            
        Returns:
            Complete results with reasoning and final diagnosis
        """
        
        # Track the single step with prompt and response
        prompts_and_responses = []
        
        try:
            print("🔍 Single-step direct reasoning: Analyzing all clinical data with all possible diagnoses...")
            
            # Get complete clinical information (all 6 inputs)
            full_summary = self.evaluator.create_patient_data_summary(sample, 6)
            
            # Create single-step direct reasoning prompt
            prompt = self._create_single_step_direct_reasoning_prompt(full_summary)
            
            # Query LLM with generous token limit for reasoning
            response = self.evaluator.query_llm(prompt, max_tokens=1500)
            
            # Parse final diagnosis from response
            final_diagnosis = self._parse_single_step_final_diagnosis(response)
            matched_diagnosis = self.evaluator.find_best_match(final_diagnosis)
            
            # Extract reasoning from response
            reasoning = self._extract_single_step_reasoning(response)
            
            # Create step record
            step_result = {
                'step': 1,
                'stage': 'single_step_direct_reasoning',
                'action': 'direct_diagnosis_with_reasoning',
                'prompt': prompt,
                'response': response,
                'extracted_diagnosis': final_diagnosis,
                'matched_diagnosis': matched_diagnosis,
                'reasoning': reasoning
            }
            prompts_and_responses.append(step_result)
            
            print(f"✅ Single-step reasoning complete: {matched_diagnosis}")
            
            return {
                'final_diagnosis': matched_diagnosis,
                'reasoning_steps': 1,
                'suspicions': [],  # Not applicable for single-step
                'recommended_tests': '',  # Not applicable for single-step
                'chosen_suspicion': matched_diagnosis,
                'reasoning_successful': bool(matched_diagnosis),
                'prompts_and_responses': prompts_and_responses,
                'reasoning_trace': prompts_and_responses,
                'mode': 'single_step_direct_reasoning'
            }
            
        except Exception as e:
            print(f"❌ Error in single-step direct reasoning: {e}")
            return self._create_failure_result(f"Single-step reasoning error: {e}", prompts_and_responses)
    
    async def run_single_step_direct_reasoning_async(self, sample: Dict, request_prefix: str = "") -> Dict:
        """
        Async version: Single-step direct reasoning with all patient info and possible diagnoses.
        
        Args:
            sample: Clinical sample data
            request_prefix: Unique prefix for tracking requests
            
        Returns:
            Complete results with reasoning and final diagnosis
        """
        
        # Track the single step with prompt and response
        prompts_and_responses = []
        
        try:
            # Get complete clinical information (all 6 inputs)
            full_summary = self.evaluator.create_patient_data_summary(sample, 6)
            
            # Create single-step direct reasoning prompt
            prompt = self._create_single_step_direct_reasoning_prompt(full_summary)
            
            # Query LLM async with generous token limit for reasoning
            response_data = await self.evaluator.query_llm_async(prompt, f"{request_prefix}_single_step", max_tokens=1500)
            
            if not response_data['success']:
                raise Exception(f"Single-step API call failed: {response_data['error']}")
            
            response = response_data['response']
            
            # Parse final diagnosis from response
            final_diagnosis = self._parse_single_step_final_diagnosis(response)
            matched_diagnosis = self.evaluator.find_best_match(final_diagnosis)
            
            # Extract reasoning from response
            reasoning = self._extract_single_step_reasoning(response)
            
            # Create step record
            step_result = {
                'step': 1,
                'stage': 'single_step_direct_reasoning',
                'action': 'direct_diagnosis_with_reasoning',
                'prompt': prompt,
                'response': response,
                'extracted_diagnosis': final_diagnosis,
                'matched_diagnosis': matched_diagnosis,
                'reasoning': reasoning,
                'request_id': f"{request_prefix}_single_step"
            }
            prompts_and_responses.append(step_result)
            
            return {
                'final_diagnosis': matched_diagnosis,
                'reasoning_steps': 1,
                'suspicions': [],  # Not applicable for single-step
                'recommended_tests': '',  # Not applicable for single-step
                'chosen_suspicion': matched_diagnosis,
                'reasoning_successful': bool(matched_diagnosis),
                'prompts_and_responses': prompts_and_responses,
                'reasoning_trace': prompts_and_responses,
                'mode': 'single_step_direct_reasoning_async'
            }
            
        except Exception as e:
            print(f"❌ Error in async single-step direct reasoning: {e}")
            return self._create_failure_result(f"Async single-step reasoning error: {e}", prompts_and_responses)


def integrate_clean_progressive_reasoning(evaluator):
    """
    Integration function to replace existing progressive reasoning workflow
    """
    
    # Create the clean progressive reasoning instance
    clean_reasoning = CleanProgressiveReasoning(evaluator)
    
    # Replace the progressive reasoning method
    def new_progressive_reasoning_workflow(self, sample: Dict, num_suspicions: int = 3,
                                         max_reasoning_steps: int = 5, fast_mode: bool = False) -> Dict:
        """Wrapper to use clean progressive reasoning"""
        return clean_reasoning.run_progressive_workflow(
            sample, num_suspicions, max_reasoning_steps
        )
    
    # Add the new single-step direct reasoning method
    def new_single_step_direct_reasoning(self, sample: Dict) -> Dict:
        """Wrapper to use single-step direct reasoning"""
        return clean_reasoning.run_single_step_direct_reasoning(sample)
    
    # Add async version of single-step direct reasoning
    async def new_single_step_direct_reasoning_async(self, sample: Dict, request_prefix: str = "") -> Dict:
        """Async wrapper to use single-step direct reasoning"""
        return await clean_reasoning.run_single_step_direct_reasoning_async(sample, request_prefix)
    
    # Bind the new methods to the evaluator
    import types
    evaluator.progressive_reasoning_workflow = types.MethodType(new_progressive_reasoning_workflow, evaluator)
    evaluator.single_step_direct_reasoning = types.MethodType(new_single_step_direct_reasoning, evaluator)
    evaluator.single_step_direct_reasoning_async = types.MethodType(new_single_step_direct_reasoning_async, evaluator)
    
    print("✅ Clean progressive reasoning (FIXED) integrated successfully!")
    print("✅ Single-step direct reasoning integrated successfully!")
    return evaluator 