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

# Import flowchart utilities (these need to exist in the main codebase)
try:
    from medeval.utils import (
        load_flowchart_content, get_flowchart_knowledge, 
        get_flowchart_structure, get_flowchart_children, 
        is_leaf_diagnosis, get_flowchart_first_step
    )
except ImportError:
    # Fallback implementations if utilities don't exist
    def load_flowchart_content(category: str, flowchart_dir: str) -> Dict:
        return {"mock": "flowchart"}
    
    def get_flowchart_knowledge(data: Dict) -> Dict:
        return {"mock": "knowledge"}
    
    def get_flowchart_structure(data: Dict) -> Dict:
        return {"mock": "structure"}
    
    def get_flowchart_children(structure: Dict, node: str) -> List[str]:
        return ["mock_child_1", "mock_child_2"]
    
    def is_leaf_diagnosis(structure: Dict, node: str) -> bool:
        return False
    
    def get_flowchart_first_step(data: Dict) -> str:
        return "Suspected Mock Disease"


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
    
    # === PROMPT CREATION METHODS ===
    
    def _create_step0_prompt(self, history_summary: str, num_candidates: int) -> str:
        """Create Step 0 prompt for choosing candidates based on history only"""
        
        prompt = f"""You are a medical expert analyzing patient history to identify initial diagnostic candidates.

**Patient History (Initial Information Only):**
{history_summary}

**Available Disease Categories:**
{chr(10).join(f'{i+1}. {cat}' for i, cat in enumerate(self.flowchart_categories))}

**Task:** Based on the patient history above, select the {num_candidates} most likely disease categories.

**Instructions:**
• Focus only on the patient history provided (no additional test results yet)
• Choose categories that best match the presenting symptoms and history
• Provide reasoning for each choice based on historical clinical findings
• Rank them in order of likelihood

**Format:**
**CLINICAL REASONING:**
[Systematic analysis of history against potential categories]

**FINAL CANDIDATES:**
1. [Most likely category with brief justification]
2. [Second most likely category with brief justification]
3. [Third most likely category with brief justification]

**DETAILED REASONING:** [Complete explanation of choice rationale]"""
        
        return prompt
    
    def _create_step1_improved_prompt(self, full_summary: str, flowchart_info: str) -> str:
        """Create Step 1 prompt for choosing from flowchart diagnostic structures"""
        
        prompt = f"""You are a medical expert with complete clinical information and access to diagnostic flowcharts.

**Complete Patient Clinical Information:**
{full_summary}

{flowchart_info}

**Task:** Choose the best initial diagnosis from the flowchart diagnostic structures above.

**CRITICAL INSTRUCTIONS:**
• You must choose from the FIRST-LEVEL diagnoses in the diagnostic structures (e.g., "Suspected Pneumonia", "Suspected Pulmonary Embolism")
• These first-level diagnoses are the entry points to the flowcharts
• Choose the initial diagnosis where the patient's clinical findings best match
• Focus on evidence-based matching between patient findings and suspected conditions
• Do NOT choose deeper-level diagnoses - only the first-level "Suspected..." diagnoses

**Analysis Required:**
• Compare patient's clinical findings against each suspected condition
• Identify which suspected diagnosis best fits the patient presentation
• Provide detailed evidence matching for your choice
• Explain why your chosen suspected diagnosis is superior to other options

**Format:**
**CLINICAL EVIDENCE ANALYSIS:**
[Systematic comparison of patient findings against each suspected condition]

**COMPARATIVE REASONING:**
[Compare how well patient matches each suspected diagnosis and explain your choice]

**CHOSEN INITIAL DIAGNOSIS:** [Exact first-level diagnosis name from the diagnostic structures above - must start with "Suspected"]
**DETAILED REASONING:** [Complete medical justification with evidence matching]"""
        
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

**Task:** Choose the most appropriate next step in the flowchart based on patient's clinical findings.

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
[Why chosen option best fits the flowchart progression and patient evidence]

**CHOSEN STEP:** [Number] - [Step name]
**REASONING:** [Complete medical justification based on flowchart criteria]"""
        
        return prompt
    
    # === PARSING METHODS ===
    
    def _parse_step0_candidates(self, response: str, num_candidates: int) -> List[str]:
        """Parse chosen candidates from Step 0 response"""
        
        candidates = []
        
        # Try to extract from FINAL CANDIDATES section
        final_match = re.search(r'FINAL CANDIDATES:\s*(.*?)(?=\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
        if final_match:
            lines = final_match.group(1).strip().split('\n')
            for line in lines:
                # Look for category names in the line - extract clean names
                clean_candidate = self._extract_clean_category_name(line)
                if clean_candidate and clean_candidate not in candidates:
                    candidates.append(clean_candidate)
        
        # Fallback: numbered lists anywhere in response
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
    
    def _extract_clean_category_name(self, text: str) -> str:
        """Extract clean category name from LLM response text"""
        
        # Remove markdown formatting
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        clean_text = clean_text.strip()
        
        # Common patterns to extract category names
        patterns = [
            r'(\d+\.\s*)?([A-Za-z][A-Za-z\s]+?)(?:\s*-|\s*\(|\Z)',  # Extract before dash or parenthesis
            r'([A-Za-z][A-Za-z\s]+?)(?:\s*-)',  # Extract before dash
            r'([A-Za-z][A-Za-z\s]+?)(?:\s*\()',  # Extract before parenthesis
            r'([A-Za-z][A-Za-z\s]+?)(?:\s*:)',  # Extract before colon
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_text)
            if match:
                candidate = match.group(1) if match.group(1) and not match.group(1).strip().endswith('.') else match.group(2)
                candidate = candidate.strip()
                
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
                    'Asthma': 'Asthma'
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
        """Load complete flowcharts for candidates and present raw diagnostic structures for selection"""
        
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
                    flowchart_info += f'"diagnostic": {diagnostic_structure}\n\n'
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
    
    # Bind the new method to the evaluator
    import types
    evaluator.progressive_reasoning_workflow = types.MethodType(new_progressive_reasoning_workflow, evaluator)
    
    print("✅ Clean progressive reasoning (FIXED) integrated successfully!")
    return evaluator 