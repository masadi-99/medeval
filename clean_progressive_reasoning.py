#!/usr/bin/env python3
"""
Clean Progressive Reasoning System - From Scratch
Based on user's exact specifications:

Step 0: History (inputs 1-4) + possible diagnoses list → Choose k=3 top candidates
Step 1: Exams/results (inputs 5-6) + k flowcharts → Choose starting diagnosis  
Step 2-n: Patient info + flowchart position → Reason to next step until leaf node

All prompts, responses, and reasoning saved for every step.
"""

from typing import Dict, List, Optional, Tuple
import re
import json

# Import flowchart utilities (these need to exist in the main codebase)
try:
    from medeval.flowchart_utils import (
        load_flowchart_content, get_flowchart_knowledge, 
        get_flowchart_structure, get_flowchart_children, 
        is_leaf_diagnosis
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


class CleanProgressiveReasoning:
    """
    Clean implementation of progressive clinical reasoning.
    Follows user's exact specifications step by step.
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
        Main entry point for clean progressive reasoning workflow.
        
        Args:
            sample: Clinical sample data
            num_candidates: Number of initial candidates (k=3)
            max_reasoning_steps: Max steps after step 1
            
        Returns:
            Complete results with all prompts/responses saved
        """
        
        # Track ALL steps with prompts and responses
        all_steps = []
        
        try:
            # STEP 0: History → Choose k candidate diagnoses
            print(f"🔍 Step 0: Analyzing history to choose {num_candidates} candidates...")
            step0_result = self._step0_choose_candidates(sample, num_candidates)
            all_steps.append(step0_result)
            
            if not step0_result.get('chosen_candidates'):
                return self._create_failure_result("Step 0 failed: No candidates chosen", all_steps)
            
            # STEP 1: Exams/results + flowcharts → Choose starting diagnosis
            print("🔍 Step 1: Analyzing complete clinical data with flowcharts...")
            step1_result = self._step1_choose_starting_diagnosis(sample, step0_result['chosen_candidates'])
            all_steps.append(step1_result)
            
            if not step1_result.get('starting_diagnosis'):
                return self._create_failure_result("Step 1 failed: No starting diagnosis", all_steps)
            
            # STEP 2-n: Flowchart-guided reasoning to final diagnosis
            print(f"🔍 Steps 2-{max_reasoning_steps + 1}: Following flowchart to final diagnosis...")
            flowchart_steps = self._steps2_n_flowchart_reasoning(
                sample, 
                step1_result['starting_diagnosis'],
                step1_result['flowchart_category'], 
                max_reasoning_steps
            )
            all_steps.extend(flowchart_steps['steps'])
            
            # Determine final diagnosis
            final_diagnosis = flowchart_steps.get('final_diagnosis') or step1_result['starting_diagnosis']
            matched_diagnosis = self.evaluator.find_best_match(final_diagnosis)
            
            print(f"✅ Workflow complete: {len(all_steps)} steps, final diagnosis: {matched_diagnosis}")
            
            return {
                'final_diagnosis': matched_diagnosis,
                'reasoning_trace': all_steps,
                'reasoning_steps': len(all_steps),
                'suspicions': step0_result['chosen_candidates'],
                'recommended_tests': step1_result.get('flowcharts_requested', ''),
                'chosen_suspicion': step1_result['starting_diagnosis'],
                'reasoning_successful': bool(matched_diagnosis),
                'prompts_and_responses': all_steps,  # Complete audit trail
                'mode': 'clean_step_by_step'
            }
            
        except Exception as e:
            print(f"❌ Error in progressive workflow: {e}")
            return self._create_failure_result(f"Workflow error: {e}", all_steps)
    
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
            'action': 'choose_candidates',
            'prompt': prompt,
            'response': response,
            'chosen_candidates': chosen_candidates,
            'history_summary': history_summary,
            'reasoning': self._extract_step0_reasoning(response)
        }
    
    def _step1_choose_starting_diagnosis(self, sample: Dict, candidates: List[str]) -> Dict:
        """
        Step 1: Exams/results (inputs 5-6) + k flowcharts → Choose starting diagnosis
        """
        
        # Get complete clinical information (all 6 inputs)
        full_summary = self.evaluator.create_patient_data_summary(sample, 6)
        
        # Load flowcharts for candidates
        flowchart_info, loaded_flowcharts = self._load_flowcharts_info(candidates)
        
        # Create Step 1 prompt
        prompt = self._create_step1_prompt(full_summary, candidates, flowchart_info)
        
        # Query LLM
        response = self.evaluator.query_llm(prompt, max_tokens=1000)
        
        # Parse starting diagnosis and flowchart
        starting_diagnosis, flowchart_category = self._parse_step1_choice(
            response, candidates, loaded_flowcharts
        )
        
        return {
            'step': 1,
            'action': 'choose_starting_diagnosis',
            'prompt': prompt,
            'response': response,
            'starting_diagnosis': starting_diagnosis,
            'flowchart_category': flowchart_category,
            'flowcharts_requested': f"Flowcharts for: {', '.join(candidates)}",
            'reasoning': self._extract_step1_reasoning(response)
        }
    
    def _steps2_n_flowchart_reasoning(self, sample: Dict, starting_diagnosis: str, 
                                     flowchart_category: str, max_steps: int) -> Dict:
        """
        Steps 2-n: Patient info + flowchart position → Reason to next step until leaf node
        """
        
        steps = []
        current_diagnosis = starting_diagnosis
        step_number = 2
        
        # Get complete clinical information
        full_summary = self.evaluator.create_patient_data_summary(sample, 6)
        
        # Load flowchart structure
        try:
            flowchart_data = load_flowchart_content(flowchart_category, self.flowchart_dir)
            flowchart_structure = get_flowchart_structure(flowchart_data)
            flowchart_knowledge = get_flowchart_knowledge(flowchart_data)
        except Exception as e:
            return {
                'steps': [{
                    'step': 2,
                    'action': 'flowchart_unavailable',
                    'response': f'Flowchart for {flowchart_category} unavailable: {e}',
                    'current_diagnosis': starting_diagnosis,
                    'final_diagnosis': True
                }],
                'final_diagnosis': starting_diagnosis
            }
        
        # Continue reasoning until leaf node or max steps
        while step_number <= max_steps + 1:  # +1 because we start at step 2
            
            # Check if current diagnosis is a leaf node (final diagnosis)
            if is_leaf_diagnosis(flowchart_structure, current_diagnosis):
                steps.append({
                    'step': step_number,
                    'action': 'reached_leaf_node',
                    'current_diagnosis': current_diagnosis,
                    'final_diagnosis': True,
                    'response': f'Reached final diagnosis: {current_diagnosis}'
                })
                break
            
            # Get next possible diagnoses from flowchart
            next_options = get_flowchart_children(flowchart_structure, current_diagnosis)
            
            if not next_options:
                steps.append({
                    'step': step_number,
                    'action': 'no_more_options',
                    'current_diagnosis': current_diagnosis,
                    'final_diagnosis': True,
                    'response': f'No further options. Final: {current_diagnosis}'
                })
                break
            
            if len(next_options) == 1:
                # Only one option, proceed automatically
                next_diagnosis = next_options[0]
                steps.append({
                    'step': step_number,
                    'action': 'single_option',
                    'current_diagnosis': current_diagnosis,
                    'next_diagnosis': next_diagnosis,
                    'response': f'Single path: {current_diagnosis} → {next_diagnosis}'
                })
                current_diagnosis = next_diagnosis
                step_number += 1
                continue
            
            # Multiple options - need LLM reasoning
            step_result = self._reason_flowchart_step(
                sample, current_diagnosis, next_options, step_number,
                full_summary, flowchart_knowledge
            )
            steps.append(step_result)
            
            # Move to chosen diagnosis
            current_diagnosis = step_result['chosen_diagnosis']
            step_number += 1
        
        return {
            'steps': steps,
            'final_diagnosis': current_diagnosis
        }
    
    def _reason_flowchart_step(self, sample: Dict, current_diagnosis: str,
                              next_options: List[str], step_number: int,
                              full_summary: str, flowchart_knowledge: Dict) -> Dict:
        """
        Reason through a flowchart step with multiple options
        """
        
        # Create prompt for this reasoning step
        prompt = self._create_flowchart_step_prompt(
            full_summary, current_diagnosis, next_options, step_number, flowchart_knowledge
        )
        
        # Query LLM
        response = self.evaluator.query_llm(prompt, max_tokens=1000)
        
        # Parse chosen diagnosis
        chosen_diagnosis = self._parse_flowchart_step_choice(response, next_options)
        
        return {
            'step': step_number,
            'action': 'flowchart_reasoning',
            'prompt': prompt,
            'response': response,
            'current_diagnosis': current_diagnosis,
            'next_options': next_options,
            'chosen_diagnosis': chosen_diagnosis,
            'reasoning': self._extract_flowchart_reasoning(response)
        }
    
    # === PROMPT CREATION METHODS ===
    
    def _create_step0_prompt(self, history_summary: str, num_candidates: int) -> str:
        """Create Step 0 prompt for choosing candidates from history"""
        
        prompt = f"""You are a medical expert analyzing patient history to identify the most likely diagnostic categories.

**Patient History (Initial Information Only):**
{history_summary}

**Available Diagnostic Categories:**"""
        
        for i, category in enumerate(self.flowchart_categories, 1):
            prompt += f"\n{i}. {category}"
        
        prompt += f"""

**Task:** Based ONLY on the patient history above, identify the {num_candidates} most likely diagnostic categories.

**Instructions:**
• Choose {num_candidates} categories from the available list above
• Rank them in order of likelihood based on historical findings
• For each choice, provide detailed reasoning based on specific historical patterns
• For rejected obvious alternatives, explain why they are less likely
• Use medical knowledge to match historical presentations with diagnostic categories

**Format:**
**REASONING FOR EACH CANDIDATE:**
1. [Category name] - [Detailed reasoning based on history]
2. [Category name] - [Detailed reasoning based on history]
3. [Category name] - [Detailed reasoning based on history]

**REJECTED ALTERNATIVES:**
- [Category name]: [Why rejected based on history]
- [Category name]: [Why rejected based on history]

**FINAL CANDIDATES:**
1. [Category name]
2. [Category name]
3. [Category name]"""
        
        return prompt
    
    def _create_step1_prompt(self, full_summary: str, candidates: List[str], 
                            flowchart_info: str) -> str:
        """Create Step 1 prompt for choosing starting diagnosis with flowcharts"""
        
        prompt = f"""You are a medical expert with complete clinical information and diagnostic flowcharts.

**Complete Patient Clinical Information:**
{full_summary}

**Available Diagnostic Flowcharts:**
{flowchart_info}

**Previously Identified Candidates (Step 0):**
{', '.join(f'{i+1}. {cand}' for i, cand in enumerate(candidates))}

**Task:** With complete clinical information and flowcharts, choose the best starting diagnosis.

**Instructions:**
• Compare complete clinical findings against each flowchart's criteria
• Choose the flowchart and starting diagnosis that best matches the patient
• Provide detailed reasoning comparing findings to flowchart requirements
• Explain why chosen option is superior to other candidates
• Identify the specific starting point in the chosen flowchart

**Format:**
**CLINICAL EVIDENCE ANALYSIS:**
[Systematic analysis of patient findings against each flowchart]

**COMPARATIVE REASONING:**
- {candidates[0]}: [How clinical findings match/don't match]
- {candidates[1] if len(candidates) > 1 else 'N/A'}: [How clinical findings match/don't match]
- {candidates[2] if len(candidates) > 2 else 'N/A'}: [How clinical findings match/don't match]

**CHOSEN FLOWCHART:** [Best matching flowchart]
**STARTING DIAGNOSIS:** [Specific starting point in flowchart]
**DETAILED REASONING:** [Complete justification]"""
        
        return prompt
    
    def _create_flowchart_step_prompt(self, full_summary: str, current_diagnosis: str,
                                     next_options: List[str], step_number: int,
                                     flowchart_knowledge: Dict) -> str:
        """Create prompt for reasoning through flowchart step"""
        
        prompt = f"""You are a medical expert following a diagnostic flowchart step-by-step.

**Complete Patient Clinical Information:**
{full_summary}

**Current Position in Flowchart:** {current_diagnosis}

**Next Possible Diagnoses:**"""
        
        for i, option in enumerate(next_options, 1):
            prompt += f"\n{i}. {option}"
        
        # Add flowchart knowledge
        if flowchart_knowledge:
            prompt += f"\n\n**Relevant Medical Knowledge:**"
            for key, value in flowchart_knowledge.items():
                if isinstance(value, dict):
                    prompt += f"\n• {key}:"
                    for subkey, subvalue in value.items():
                        prompt += f"\n  - {subkey}: {subvalue}"
                else:
                    prompt += f"\n• {key}: {value}"
        
        prompt += f"""

**Task:** Choose the most appropriate next diagnosis based on patient's clinical findings.

**Instructions:**
• Compare patient's findings against each option's typical presentation
• Use medical knowledge to guide reasoning
• Provide detailed evidence matching for each option
• Explain why chosen option best fits the clinical picture
• Explain why other options are less likely

**Format:**
**EVIDENCE MATCHING:**
1. {next_options[0]}: [How patient findings match/don't match]
{f'2. {next_options[1]}: [How patient findings match/don\'t match]' if len(next_options) > 1 else ''}
{f'3. {next_options[2]}: [How patient findings match/don\'t match]' if len(next_options) > 2 else ''}

**COMPARATIVE ANALYSIS:**
[Why chosen option is superior to alternatives]

**CHOSEN DIAGNOSIS:** [Number] - [Diagnosis name]
**REASONING:** [Complete medical justification]"""
        
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
                match = re.match(r'^\d+\.\s*(.+)', line.strip())
                if match:
                    candidate = match.group(1).strip()
                    if candidate and candidate not in candidates:
                        candidates.append(candidate)
        
        # Fallback: numbered lists
        if len(candidates) < num_candidates:
            all_matches = re.findall(r'^\d+\.\s*([A-Za-z][A-Za-z\s]+)', response, re.MULTILINE)
            for match in all_matches:
                clean_match = match.strip()
                if clean_match and clean_match not in candidates and len(clean_match) > 3:
                    candidates.append(clean_match)
                    if len(candidates) >= num_candidates:
                        break
        
        # Ensure correct number
        while len(candidates) < num_candidates:
            candidates.append(f"Unknown Category {len(candidates) + 1}")
        
        return candidates[:num_candidates]
    
    def _parse_step1_choice(self, response: str, candidates: List[str],
                           loaded_flowcharts: Dict) -> Tuple[str, str]:
        """Parse starting diagnosis choice from Step 1 response"""
        
        # Extract flowchart and starting diagnosis
        flowchart_match = re.search(r'CHOSEN FLOWCHART:\s*(.+?)(?=\n|\Z)', response, re.IGNORECASE)
        diagnosis_match = re.search(r'STARTING DIAGNOSIS:\s*(.+?)(?=\n|\Z)', response, re.IGNORECASE)
        
        chosen_flowchart = None
        starting_diagnosis = None
        
        if flowchart_match:
            flowchart_text = flowchart_match.group(1).strip()
            for candidate in candidates:
                if candidate.lower() in flowchart_text.lower():
                    chosen_flowchart = candidate
                    break
        
        if diagnosis_match:
            starting_diagnosis = diagnosis_match.group(1).strip()
        
        # Fallbacks
        if not chosen_flowchart:
            chosen_flowchart = candidates[0]
        if not starting_diagnosis:
            starting_diagnosis = chosen_flowchart
        
        return starting_diagnosis, chosen_flowchart
    
    def _parse_flowchart_step_choice(self, response: str, options: List[str]) -> str:
        """Parse chosen diagnosis from flowchart step response"""
        
        # Extract from CHOSEN DIAGNOSIS section
        chosen_match = re.search(r'CHOSEN DIAGNOSIS:\s*(\d+)\s*-\s*(.+?)(?=\n|\Z)', 
                               response, re.IGNORECASE)
        
        if chosen_match:
            try:
                choice_num = int(chosen_match.group(1))
                if 1 <= choice_num <= len(options):
                    return options[choice_num - 1]
            except ValueError:
                pass
        
        # Fallback: any number
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
    
    def _load_flowcharts_info(self, candidates: List[str]) -> Tuple[str, Dict]:
        """Load flowchart information for candidates"""
        
        flowchart_info = ""
        loaded_flowcharts = {}
        
        for candidate in candidates:
            try:
                flowchart_data = load_flowchart_content(candidate, self.flowchart_dir)
                flowchart_knowledge = get_flowchart_knowledge(flowchart_data)
                loaded_flowcharts[candidate] = flowchart_data
                
                flowchart_info += f"\n**{candidate} Flowchart:**\n"
                if flowchart_knowledge:
                    for key, value in flowchart_knowledge.items():
                        if isinstance(value, dict):
                            flowchart_info += f"• {key}:\n"
                            for subkey, subvalue in value.items():
                                flowchart_info += f"  - {subkey}: {subvalue}\n"
                        else:
                            flowchart_info += f"• {key}: {value}\n"
                
            except Exception as e:
                flowchart_info += f"\n**{candidate} Flowchart:** [Could not load: {e}]\n"
        
        return flowchart_info, loaded_flowcharts
    
    def _extract_step0_reasoning(self, response: str) -> str:
        """Extract reasoning from Step 0 response"""
        reasoning_match = re.search(r'REASONING FOR EACH CANDIDATE:\s*(.*?)(?=\n\*\*REJECTED|\n\*\*FINAL|\Z)', 
                                  response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            return reasoning_match.group(1).strip()
        return response[:200] + "..." if len(response) > 200 else response
    
    def _extract_step1_reasoning(self, response: str) -> str:
        """Extract reasoning from Step 1 response"""
        reasoning_match = re.search(r'DETAILED REASONING:\s*(.*?)(?=\n\*\*|\Z)', 
                                  response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            return reasoning_match.group(1).strip()
        return response[:200] + "..." if len(response) > 200 else response
    
    def _extract_flowchart_reasoning(self, response: str) -> str:
        """Extract reasoning from flowchart step response"""
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?=\n\*\*|\Z)', 
                                  response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            return reasoning_match.group(1).strip()
        return response[:200] + "..." if len(response) > 200 else response
    
    def _create_failure_result(self, error_message: str, steps: List[Dict]) -> Dict:
        """Create failure result when workflow fails"""
        return {
            'final_diagnosis': None,
            'reasoning_trace': steps,
            'reasoning_steps': len(steps),
            'suspicions': [],
            'recommended_tests': '',
            'chosen_suspicion': None,
            'reasoning_successful': False,
            'prompts_and_responses': steps,
            'mode': 'clean_step_by_step',
            'error': error_message
        }


# Integration function for main evaluator
def integrate_clean_progressive_reasoning(evaluator):
    """
    Replace the existing progressive reasoning workflow with the clean implementation
    """
    
    # Create clean reasoning instance
    clean_reasoning = CleanProgressiveReasoning(evaluator)
    
    # Replace the existing method with proper signature
    def new_progressive_reasoning_workflow(self, sample: Dict, num_suspicions: int = 3,
                                         max_reasoning_steps: int = 5, fast_mode: bool = False) -> Dict:
        """Clean progressive reasoning workflow - ignores fast_mode, always uses step-by-step"""
        return clean_reasoning.run_progressive_workflow(sample, num_suspicions, max_reasoning_steps)
    
    # Replace the method properly
    import types
    evaluator.progressive_reasoning_workflow = types.MethodType(new_progressive_reasoning_workflow, evaluator)
    
    print("✅ Clean progressive reasoning system integrated!")
    return evaluator 