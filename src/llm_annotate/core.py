import os
import re
import ast
import json
import itertools
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import tkinter as tk
import tiktoken

from openai import OpenAI

client = OpenAI()

def rate_evidence(annotation_file, outputfile, chunkfile, labels=["weak", "moderate", "strong"], skip_zeros=True, model="gpt-4.1"):
    """Rates evidence and writes results; parses model output into two keys with fallbacks."""
    with open(annotation_file, "r", encoding="utf-8") as f:
        action_annotations_per_character = json.load(f)
    with open(chunkfile, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    rated_annotations = {}
    for character, annotations in action_annotations_per_character.items():
        rated_annotations[character] = []
        for annotation in annotations:
            action = annotation.get("Action", "")
            chunk = chunks.get(str(annotation.get("Chunk", "")), "")
            trait = next((k for k in annotation.keys() if k not in ["Action", "Chunk"]), None)
            if not trait:
                continue
            trait_value = annotation[trait]
            if skip_zeros and trait_value == 0:
                continue
            low_high = "high" if trait_value == 1 else "low"
            prompt = f"""Previously, you noted that an action committed by {character} indicated a {low_high} level of {trait}. 
Given the original text (see below), rate how strong the evidence is for this inference.
<Character> {character} </Character>
<Action> {action} </Action>
<Text segment> \"{chunk}\" </Text segment>
<Instruction> Consider how indicative the action by {character} is for the trait {trait}.
Rate the indicativeness by responding with exactly this dictionary format: {{"Thoughts": Your consideration of the action with respect to {trait}, "Label": *Your label choice*}}. 
For the label, choose exactly one from the following options: {', '.join(labels)}.
</Instruction>"""
            response = call_model(prompt=prompt, temperature=0, model=model)
            print(prompt); print("\n\n\n"); print(response); print("-----")
            # Parse response into two separate keys
            raw = response.strip() if isinstance(response, str) else ""
            thoughts = ""
            label = ""
            if raw:
                try:
                    obj = json.loads(raw)
                    thoughts = (obj.get("Thoughts") or "").strip()
                    label = (obj.get("Label") or "").strip()
                except Exception:
                    # Fallback regex extraction
                    t_m = re.search(r'"?Thoughts"?\s*:\s*"?([^"}]+)"?', raw, re.IGNORECASE)
                    l_m = re.search(r'"?Label"?\s*:\s*"?([^"}]+)"?', raw, re.IGNORECASE)
                    if t_m:
                        thoughts = t_m.group(1).strip()
                    if l_m:
                        label = l_m.group(1).strip()
            rated_annotation = annotation.copy()
            rated_annotation["Evidence_thoughts"] = thoughts
            rated_annotation["Evidence_label"] = label
            rated_annotation["Evidence_raw"] = raw  # optional raw for auditing
            rated_annotations[character].append(rated_annotation)
    with open(outputfile, "w", encoding="utf-8") as f:
        json.dump(rated_annotations, f, ensure_ascii=False, indent=2)



def custom_openai(prompt,model_they_gave_you_access_to = "gpt-4.1-mini"):
    """Call UVA OpenAI proxy."""
    your_api_key = os.getenv("UVA_OPENAI_API_KEY")
    the_base_url_to_always_use = "https://ai-research-proxy.azurewebsites.net/"

    client = OpenAI(api_key=your_api_key, base_url=the_base_url_to_always_use)
    try:
        response = client.chat.completions.create(
        model=model_they_gave_you_access_to,
        messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except Exception as e:
        # Inspect exception text for Azure/OpenAI content-filtering indicators.
        msg = str(e).lower()
        if (
            "contentpolicyviolation" in msg
            or "content policy" in msg
            or "filtered due to" in msg
            or "content_policy" in msg
            or "contentmanagementpolicy" in msg
            or "azureexception" in msg
            or "contentpolicy" in msg
            or "litellm.contentpolicyviolationerror" in msg
        ):
            print("WARNING: The request was blocked by Azure/OpenAI content policy. Returning empty response.")
            return "[]" 
        raise ValueError(f"Error calling OpenAI API: {e}")
        
# from mistralai import Mistral
# def mistral_llm(prompt):
#     """Call Mistral API with mistral-large-latest model."""
#     client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
#     response = client.chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": prompt}])
#     return response.choices[0].message.content


def call_model(model, prompt, temperature=0):
    """Call an OpenAI model (string) or custom callable function with a prompt."""
    if isinstance(model, str):
        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature)
        return response.choices[0].message.content
    elif callable(model):
        return model(prompt)
    else:
        raise ValueError("model must be a string or a callable function")
    

def get_character_embeddings(
    annotfile,
    outputfile,
    model="text-embedding-3-small",
    min_actions=20,
    min_trait_count=1
):
    """Generate character embeddings by averaging trait embeddings from annotations.
    
    Args:
        annotfile: Path to JSON file with character annotations
        outputfile: Path to save character embeddings JSON
        model: OpenAI embedding model name
        min_actions: Minimum actions required per character
        min_trait_count: Minimum occurrences required per trait
    """
    with open(annotfile, "r") as f:
        annotations = json.load(f)

    # Count trait occurrences across all actions
    trait_counts = defaultdict(int)
    for actions in annotations.values():
        for a in actions:
            for k in a.keys():
                if k not in ("Action", "Chunk"):
                    trait_counts[k] += 1

    # Only keep traits that appear at least min_trait_count times
    valid_traits = {trait for trait, count in trait_counts.items() if count >= min_trait_count}
    print(f"nr of valid traits (appearing at least {min_trait_count} times): {len(valid_traits)}")
    valid_traits_list = list(valid_traits)
    if not valid_traits_list:
        with open(outputfile, "w") as f:
            json.dump({}, f)
        return

    # Embed each valid trait only once
    response = client.embeddings.create(
        model=model,
        input=valid_traits_list
    )
    trait_to_embedding = {trait: item.embedding for trait, item in zip(valid_traits_list, response.data)}

    # Only keep characters with at least min_actions actions
    character_trait_embeddings = {}
    for character, actions in annotations.items():
        if len(actions) < min_actions:
            continue
        # Get traits for this character that are valid
        trait_names = []
        for a in actions:
            trait_names.extend([k for k in a.keys() if k in valid_traits])
        if not trait_names:
            continue
        print(f"Character {character} has traits: {sorted(trait_names)}")
        trait_embeds = [trait_to_embedding[t] for t in trait_names]
        avg_embedding = np.mean(trait_embeds, axis=0).tolist()
        character_trait_embeddings[character] = avg_embedding

    with open(outputfile, "w") as f:
        json.dump(character_trait_embeddings, f)


def chunk_text(novel_text, outputfile, nr_chunks=None, chunk_size=500, custom_splitter=None, keep_custom_splitter=False, verbose=True):
    """Split novel text into chunks and save to JSON file.
    
    Args:
        novel_text: Full text to chunk
        outputfile: Path to save chunks JSON
        nr_chunks: Optional limit on number of chunks
        chunk_size: Approximate tokens per chunk (if no custom_splitter)
        custom_splitter: Optional string to split on
        keep_custom_splitter: Whether to keep splitter in chunks
    
    Returns:
        List of chunk strings
    """

    if custom_splitter:
        # Very simple custom splitter: split on the custom_splitter string
        chunks = novel_text.split(custom_splitter)
        if not keep_custom_splitter:
            # Remove the splitter from the start of each chunk except the first
            chunks = [chunks[0]] + [c.lstrip() for c in chunks[1:]]
        # Optionally limit number of chunks
        if nr_chunks is not None:
            chunks = chunks[:nr_chunks]
        chunk_dict = {str(idx+1): chunk for idx, chunk in enumerate(chunks)}
    else:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(novel_text)
        chunks = []
        i = 0
        while i < len(tokens):
            if nr_chunks is not None and len(chunks) >= nr_chunks:
                break

            chunk_tokens = tokens[i:i+chunk_size]
            chunk_text = encoding.decode(chunk_tokens)
            last_period = chunk_text.rfind('.')
            if last_period != -1 and i + chunk_size < len(tokens):
                chunk_text = chunk_text[:last_period+1]
                next_chunk_start_text = encoding.encode(chunk_text)
                i += len(next_chunk_start_text)
            else:
                i += chunk_size
            chunks.append(chunk_text)

        #add last sentences of previous chunk to current chunk (except for first chunk)
        for j in range(1, len(chunks)):
            #split sentences by . ! ?
            sentences = re.split(r'(?<=[.!?])\s*(?=[A-Z]|\[|\n|$)', chunks[j-1])
            sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences
            if len(sentences) > 3:
                last_sentences = " ".join(sentences[-3:])
                chunks[j] = f"""END OF PREVIOUSLY ANNOTATED SECTION:\n{last_sentences} \n\nTARGET SECTION TO BE ANNOTATED:\n{chunks[j]}"""
        chunk_dict = {str(idx+1): chunk for idx, chunk in enumerate(chunks)}
    with open(outputfile, "w", encoding="utf-8") as f:
        json.dump(chunk_dict, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"Saved {len(chunks)} chunks to {outputfile}")


def disambiguate(annotfile, new_annotation_file=None, chunkfile = None, model="gpt-4o", list_of_pseudonym_lists = [], book_title = 'Unknown title'):
    """Merge character annotations for detected or provided pseudonyms.
    
    Args:
        annotfile: Path to character annotations JSON
        new_annotation_file: Output path for disambiguated annotations
        chunkfile: Path to text chunks JSON for context
        model: LLM model for pseudonym detection
        list_of_pseudonym_lists: Optional manual pseudonym groups
        book_title: Book title for context
    
    Returns:
        Path to new annotation file
    """
    # Load character names from the annotation file
    with open(annotfile, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    if new_annotation_file is None:
        new_annotation_file = annotfile.replace(".json", "_disambiguated.json")

    if not list_of_pseudonym_lists: #ie if human didnt provide list of pseudonym lists, we need AI to look for pseudonyms

        character_names = list(annotations.keys())

        # Also load novel chunks to provide context (if needed)
        with open(chunkfile, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        novel_text = "\n".join(chunks.values())

        #get top 4 characters by number of annotations
        char_counts = {name: len(annots) for name, annots in annotations.items()}
        sorted_character_names = sorted(char_counts, key=char_counts.get, reverse=True)
        main_character_names = sorted_character_names[:min(6, len(sorted_character_names))]
        considered_pseudonyms = sorted_character_names[:min(30, len(sorted_character_names))]
        
        pseudonym_lists = []
        for char in main_character_names:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(novel_text)
            novel_sections = [encoding.decode(tokens[i:i+8000]) for i in range(0, len(tokens), 8000)]
            pseudonyms = []
            # Select up to 4 equally spaced sections from the novel
            num_sections = min(4, len(novel_sections))
            indices = np.linspace(0, len(novel_sections) - 1, num_sections, dtype=int)
            for i in indices:
                section = novel_sections[i]
                prompt = (
                    "You are a literary expert specializing in character names. "
                    f"Your task is to analyze the book {book_title} and list how the character '{char}' is referred to throughout the story. "
                    f"Pseudonyms only count if they are part of this list: {character_names}"
                    f"<Book Excerpt>\n...{section}...\n</Book Excerpt>\n\n"
                    f"<Instructions> Return a python list with aliases, nicknames, or alternate forms of the name {char}."
                    f"You can only pick from these options: {considered_pseudonyms}"
                    f"Return only a Python list with zero or more of the allowed pseudonyms. Leave out uncertain cases or merely related names. Only include VERY LIKELY pseudonyms of {char}</Instructions>\n\n"
                )
                result_text = call_model(model, prompt, temperature=0)

                # Extract the Python list of lists from the response
                match = re.search(r'\[(.*)\]', result_text, re.DOTALL)
                if match:
                    # Try to safely evaluate the list
                    try:
                        character_pseudonyms = eval("[" + match.group(1) + "]")

                    except Exception:
                        pseudonyms = []
                else:
                    pseudonyms = []
                pseudonyms.extend(character_pseudonyms)
            pseudonyms = list(set(pseudonyms))
            #restrict to candidate pseudonyms
            characters_lower = [p.lower() for p in sorted_character_names]
            pseudonyms = [p for p in pseudonyms if p.lower() in characters_lower]

            if char not in pseudonyms: 
                pseudonyms.insert(0, char)
            pseudonym_lists.append(pseudonyms)

        pseudonym_lists = [l for l in pseudonym_lists if len(l) > 1] #only keep lists with more than one name

        # Build annotation lists per name
        name_to_annots = {name: annotations.get(name, []) for group in pseudonym_lists for name in group}

        verified_pairs = set()
        for i, name_group in enumerate(pseudonym_lists):
            name_pairs = list(itertools.combinations(name_group, 2))
            for name1, name2 in name_pairs:
                annots1 = name_to_annots.get(name1, [])
                annots2 = name_to_annots.get(name2, [])
                # Determine which name is more common
                if len(annots1) >= len(annots2):
                    more_common, less_common = name1, name2
                    more_annots, less_annots = annots1, annots2
                else:
                    more_common, less_common = name2, name1
                    more_annots, less_annots = annots2, annots1
                # Sample up to two annotations for less common name
                sampled_annots = np.random.choice(less_annots, min([2, len(less_annots)]), replace=False)
                all_match = True
                for annot in sampled_annots:
                    # annot should have 'Action' and 'Chunk' keys
                    chunk_idx = annot.get("Chunk")
                    action = annot.get("Action")
                    # Get context: chunk and neighbors
                    chunk_keys = list(chunks.keys())
                    try:
                        idx = chunk_keys.index(str(chunk_idx))
                    except ValueError:
                        idx = None
                    context_chunks = []
                    if idx is not None:
                        for offset in [-1, 0, 1]:
                            neighbor_idx = idx + offset
                            if 0 <= neighbor_idx < len(chunk_keys):
                                context_chunks.append(chunks[chunk_keys[neighbor_idx]])
                    context_text = "\n".join(context_chunks)
                    # Compose prompt for OpenAI
                    prompt = (
                        f"<Excerpt>\n{context_text}\n</Excerpt>\n\n"
                        f"<Instruction>You are a pseudonym finder. Your task is to determine if the action '{action}' described in the excerpt can be attributed to a given character.\n"
                        f"In the excerpt, the action is attributed to '{less_common}'. Is this the same person as '{more_common}' (another name appearing in the full story)?</Instruction>"
                        "Return only 'Yes' or 'No'.</Instruction>"
                    )
                    result_text = call_model(model, prompt, temperature=0)
                    
                    if not "yes" in result_text.lower():
                        all_match = False
                        break
                if all_match:
                    verified_pairs.add(frozenset([name1, name2]))
                    #write to trash txt file

        # Merge overlapping pairs into groups
        groups = []
        for pair in verified_pairs:
            added = False
            for group in groups:
                if not pair.isdisjoint(group):
                    group.update(pair)
                    added = True
                    break
            if not added:
                groups.append(set(pair))

        element_counts = defaultdict(int)
        for group in groups:
            for elem in group:
                element_counts[elem] += 1
        overlapping_elements = {elem for elem, count in element_counts.items() if count > 1}
        # Remove overlapping elements from all groups
        cleaned_groups = []
        for group in groups:
            cleaned_group = set([elem for elem in group if elem not in overlapping_elements])
            if cleaned_group:
                cleaned_groups.append(cleaned_group)
        list_of_pseudonym_lists = [list(group) for group in cleaned_groups]

    #continue with list of pseudonym lists (either from AI or human user)
    for group in list_of_pseudonym_lists:
        # Find longest name in the group
        if not group:
            continue
        main_name = max(group, key=len)
        # Merge annotations from other names
        merged_annots = annotations.get(main_name, []).copy()
        for name in group:
            if name == main_name:
                continue
            merged_annots.extend(annotations.get(name, []))
            if name in annotations:
                del annotations[name]
        annotations[main_name] = merged_annots

    if list_of_pseudonym_lists:
        print("Merged character names:")
        for group in list_of_pseudonym_lists:
            print(", ".join(group))

    with open(new_annotation_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)


def compute_annotation_statistics(annotation_file, plot = True, nr_characters_plotted = 4, outputfile="character_statistics.json"):
    """Compute Bayesian statistics and credible intervals for character traits.
    
    Args:
        annotation_file: Path to annotations JSON
        plot: Whether to generate plots
        nr_characters_plotted: Number of top characters to plot
        outputfile: Path to save statistics JSON
    """
    from scipy.stats import beta
    #check if action_annotations_per_character is a filename or a dict
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    
    statistics = {}
    credible_intervals = {}

    # Extract trait names from the annotation dicts
    trait_names = set()
    for actions in annotations.values():
        for a in actions:
            trait_names.update([k for k in a.keys() if k not in ("Action", "Chunk")])
    trait_names = list(trait_names)

    for name, actions in annotations.items():
        statistics[name] = {}
        credible_intervals[name] = {}
        for trait in trait_names:
            trait_key = trait
            trait_values = [a[trait_key] for a in actions if trait_key in a]
            try:
                avg_trait = sum(trait_values) / len(trait_values) if trait_values else 0
            except:
                print(trait_values)
                raise ValueError(f"eror")
            # Bayesian credible interval (beta distribution scaled from -1 to 1)
            mapped = [(v+1)/2 for v in trait_values]
            alpha = 1 + sum(mapped)
            beta_param = 1 + len(mapped) - sum(mapped)
            lower = beta.ppf(0.025, alpha, beta_param)
            upper = beta.ppf(0.975, alpha, beta_param)
            lower = lower*2 - 1
            upper = upper*2 - 1
            statistics[name][trait] = avg_trait
            statistics[name][f"{trait}_n_positive"] = sum(1 for v in trait_values if v == 1)
            statistics[name][f"{trait}_n_negative"] = sum(1 for v in trait_values if v == -1)
            statistics[name][f"{trait}_n_neutral"] = sum(1 for v in trait_values if v == 0)
            statistics[name][f"{trait}_lower"] = lower
            statistics[name][f"{trait}_upper"] = upper
            statistics[name][f"{trait}_n"] = len(trait_values)
            statistics[name][f"{trait}_alpha"] = alpha
            statistics[name][f"{trait}_beta_param"] = beta_param

    #sort characters by number of annotations
    statistics = dict(sorted(statistics.items(), key=lambda item: sum(item[1][f"{trait}_n"] for trait in trait_names), reverse=True))

    #save to file
    with open(outputfile, "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    
    if plot:

        #if more than 10 unique traits across characters
        if len(trait_names) > 10:
            plot_character_trait_profiles(annotation_file, top_n_chars=nr_characters_plotted)
        else:
            # Plot only the top nr_characters_plotted characters by annotation count
            top_characters = list(statistics.keys())[:nr_characters_plotted]
            for trait in trait_names:
                plt.figure(figsize=(10, 6))
                x = np.linspace(-1, 1, 200)
                for name in top_characters:
                    alpha = statistics[name][f"{trait}_alpha"]
                    beta_param = statistics[name][f"{trait}_beta_param"]
                    # Transform x from [-1,1] to [0,1] for beta pdf
                    x_beta = (x + 1) / 2
                    y = beta.pdf(x_beta, alpha, beta_param)
                    plt.plot(x, y, label=name)
                plt.title(f"{trait.capitalize()} of characters")
                plt.xlabel(f"{trait.capitalize()}")
                plt.ylabel("Density")
                plt.axvline(0, color='gray', linestyle='--')
                plt.legend(title="Character")
                plt.tight_layout()
                plt.show()


def plot_character_trait_profiles(annotations_file_path, top_n_chars=4):
    """Plot trait frequency profiles for top characters.
    
    Args:
        annotations_file_path: Path to annotations JSON
        top_n_chars: Number of top characters to display
    """
    # Load the trimmed annotations
    with open(annotations_file_path, 'r', encoding='utf-8') as f:
        character_annotations_json = json.load(f)
    
    # Extract traits for each character
    character_traits = {}
    for character, actions in character_annotations_json.items():
        traits_list = []
        for action in actions:
            for key, value in action.items():
                if key not in ['Action', 'Chunk'] and value == 1:
                    traits_list.append(key)
        character_traits[character] = traits_list

    # Sort all characters by total number of traits and take top N
    sorted_all_characters = sorted(character_traits.items(), key=lambda x: len(x[1]), reverse=True)
    sorted_characters = sorted_all_characters[:top_n_chars]
    
    if sorted_characters:
        # Create a single compact plot showing top traits for all characters
        fig, ax = plt.subplots(1, 1, figsize=(14, len(sorted_characters) * 2))
        
        y_positions = []
        character_names = []
        
        for i, (character, traits) in enumerate(sorted_characters):
            trait_counts = Counter(traits)
            # Sort traits by count (descending)
            top_traits = dict(sorted(trait_counts.items(), key=lambda x: x[1], reverse=True))
            
            # Create trait text with frequencies
            trait_text = ", ".join([f"{trait} ({count})" for trait, count in top_traits.items()])
            
            # Position for this character with more spacing
            y_pos = (len(sorted_characters) - i - 1) * 1.5
            y_positions.append(y_pos)
            character_names.append(character)
            
            # Add character name (bold)
            ax.text(0.02, y_pos + 0.15, f"{character} ({len(traits)} total):", 
                    fontweight='bold', fontsize=12, va='bottom')
            
            # Add traits text (wrapped for readability)
            ax.text(0.02, y_pos - 0.15, trait_text, 
                    fontsize=10, va='top', wrap=True, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
        
        # Style the plot
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, (len(sorted_characters) - 1) * 1.5 + 0.5)
        ax.set_title('Character Trait Profiles', fontweight='bold', fontsize=16, pad=20)
        
        # Remove all axes elements
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        plt.show()


def score_annotations(annotation_file, outputfile, chunkfile, summaryfile=None, num_annotations=100, labels=None, seed=42):
    """GUI for human evaluation of annotation quality with customizable labels.
    
    Args:
        annotation_file: Path to annotations JSON
        outputfile: Path to save evaluation results
        chunkfile: Path to text chunks JSON for context
        summaryfile: Path to save summary statistics
        num_annotations: Number of annotations to evaluate
        labels: Custom label options (default: Correct/Questionable/Incorrect)
        seed: Random seed for sampling
    """
    if labels is None:
        labels = ["Correct", "Questionable", "Incorrect"]
    
    if summaryfile is None:
        summaryfile = outputfile.replace(".json", "_summary.json")
    with open(annotation_file, "r", encoding="utf-8") as f:
        action_annotations_per_character = json.load(f)
    
    # Load chunks for context display
    with open(chunkfile, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Flatten annotations
    all_annotations = []
    for character, actions in action_annotations_per_character.items():
        for action in actions:
            all_annotations.append((character, action))
    
    #print number of annotations
    print(f"Scoring {num_annotations} out of {len(all_annotations)} total annotations.")
    
    # Randomly sample annotations
    np.random.seed(seed)
    if len(all_annotations) > num_annotations:
        sampled_indices = np.random.choice(len(all_annotations), size=num_annotations, replace=False)
        annotations = [all_annotations[i] for i in sampled_indices]
    else:
        annotations = all_annotations

    # Check if output file exists and load previous results
    result = []
    start_idx = 0
    if os.path.exists(outputfile):
        print(f"Found existing output file: {outputfile}")
        try:
            with open(outputfile, "r") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            entry = json.loads(line.strip())
                            result.append(entry)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse line {line_num} in {outputfile}: {e}")
            start_idx = len(result)
            print(f"Resuming from annotation {start_idx + 1} of {len(annotations)}")
            
            # Verify that we haven't already completed all annotations
            if start_idx >= len(annotations):
                print("All annotations have already been completed!")
                return
                
        except Exception as e:
            print(f"Error reading existing output file: {e}")
            print("Starting from the beginning...")
            result = []
            start_idx = 0

    # --- Enhanced Tkinter GUI with simplified styling ---
    root = tk.Tk()
    root.title("📚 Character Annotation Evaluator")
    root.geometry("1200x900")
    root.configure(bg='#f8f9fa')
    
    # Configure style
    root.option_add('*Font', 'Segoe\ UI 10')

    idx = [start_idx]  # Start from where we left off
    
    # Header frame with title and progress
    header_frame = tk.Frame(root, bg='#495057', height=80)
    header_frame.pack(fill=tk.X, padx=0, pady=0)
    header_frame.pack_propagate(False)

    title_label = tk.Label(
        header_frame, 
        text="📚 Character Annotation Evaluator", 
        font=("Segoe UI", 18, "bold"), 
        fg='white', 
        bg='#495057'
    )
    title_label.pack(pady=(15, 5))

    progress_label = tk.Label(
        header_frame, 
        text=f"Annotation 1 of {len(annotations)}", 
        font=("Segoe UI", 12), 
        fg='#e9ecef', 
        bg='#495057'
    )
    progress_label.pack()

    # Main content frame
    content_frame = tk.Frame(root, bg='#f8f9fa')
    content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))

    # Combined character and annotation info frame
    annot_frame = tk.Frame(content_frame, bg='#6c757d', relief=tk.RAISED, bd=2)
    annot_frame.pack(fill=tk.X, pady=(0, 15))

    annot_header = tk.Label(
        annot_frame, 
        text="📝 AI annotation", 
        font=("Segoe UI", 12, "bold"), 
        fg='white', 
        bg='#6c757d',
        pady=5
    )
    annot_header.pack()

    annot_text = tk.Text(
        annot_frame, 
        height=8, 
        font=("Consolas", 12), 
        bg='#ffffff', 
        fg='#212529',
        relief=tk.FLAT,
        padx=10,
        pady=5
    )
    annot_text.pack(fill=tk.X, padx=10, pady=(0, 10))

    # Chunk content frame - simplified colors
    chunk_frame = tk.Frame(content_frame, bg='#6c757d', relief=tk.RAISED, bd=2)
    chunk_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

    chunk_header = tk.Label(
        chunk_frame, 
        text="📖 Original Text Context", 
        font=("Segoe UI", 12, "bold"), 
        fg='white', 
        bg='#6c757d',
        pady=5
    )
    chunk_header.pack()

    # Search frame
    search_frame = tk.Frame(chunk_frame, bg='#6c757d')
    search_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
    
    search_var = tk.StringVar()
    search_entry = tk.Entry(
        search_frame,
        textvariable=search_var,
        font=("Segoe UI", 10),
        bg='#ffffff',
        fg='#212529',
        relief=tk.FLAT,
        width=30
    )
    search_entry.pack(side=tk.RIGHT, padx=(10, 0))
    
    search_label = tk.Label(
        search_frame,
        text="🔍 Search:",
        font=("Segoe UI", 10),
        fg='white',
        bg='#6c757d'
    )
    search_label.pack(side=tk.RIGHT, padx=(0, 5))

    # Scrollable text widget for chunk content
    text_container = tk.Frame(chunk_frame, bg='#6c757d')
    text_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    
    text_widget = tk.Text(
        text_container, 
        wrap=tk.WORD, 
        font=("Georgia", 13), 
        bg='#ffffff', 
        fg='#212529',
        relief=tk.FLAT,
        padx=15,
        pady=10,
        selectbackground='#007bff',
        selectforeground='white'
    )
    scrollbar = tk.Scrollbar(text_container, orient=tk.VERTICAL, command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Configure text widget tags for highlighting
    text_widget.tag_configure("char_bold", font=("Georgia", 13, "bold"), foreground="#007bff")
    text_widget.tag_configure("search_highlight", background="#ffff00", foreground="#000000")

    def search_text():
        """Search for text in the text widget and highlight matches."""
        search_term = search_var.get().strip()
        
        # Remove previous search highlights
        text_widget.tag_remove("search_highlight", "1.0", tk.END)
        
        if not search_term:
            return
        
        # Search for all occurrences of the search term (case-insensitive)
        start_pos = "1.0"
        while True:
            pos = text_widget.search(search_term, start_pos, tk.END, nocase=True)
            if not pos:
                break
            
            # Calculate end position
            end_pos = f"{pos}+{len(search_term)}c"
            
            # Highlight the match
            text_widget.tag_add("search_highlight", pos, end_pos)
            
            # Move to next character for next search
            start_pos = f"{pos}+1c"
    
    # Bind search function to entry changes
    search_var.trace('w', lambda *args: search_text())
    
    # Bind Enter key to search entry for convenience
    search_entry.bind('<Return>', lambda e: search_text())

    # Buttons frame with enhanced styling
    btn_frame = tk.Frame(root, bg='#495057', height=80)
    btn_frame.pack(fill=tk.X, pady=0)
    btn_frame.pack_propagate(False)

    btn_container = tk.Frame(btn_frame, bg='#495057')
    btn_container.pack(expand=True)

    def update_annotation():
        if idx[0] < len(annotations):
            character, action = annotations[idx[0]]
            
            # Update progress
            progress_label.config(text=f"Annotation {idx[0] + 1} of {len(annotations)}")
            
            # Clear search when updating annotation
            search_var.set("")
            
            # Update annotation details including character name with bold formatting
            annot_text.config(state=tk.NORMAL)
            annot_text.delete(1.0, tk.END)
            
            # Configure bold tag
            annot_text.tag_configure("bold", font=("Consolas", 12, "bold"))
            
            # Insert character name with bold formatting
            annot_text.insert(tk.END, "Character: ", "bold")
            annot_text.insert(tk.END, f"{character}\n")
            
            # Insert other annotation details, excluding chunk/evidence keys here
            for key, value in action.items():
                if key in ["Chunk", "Evidence_raw", "Evidence_thoughts", "Evidence_label"]:
                    continue
                annot_text.insert(tk.END, f"{key}: ", "bold")
                annot_text.insert(tk.END, f"{value}\n")

            # Evidence section (if present)
            has_evidence_label = "Evidence_label" in action and action["Evidence_label"]
            has_evidence_thoughts = "Evidence_thoughts" in action and action["Evidence_thoughts"]
            if has_evidence_label or has_evidence_thoughts:
                if has_evidence_label:
                    annot_text.insert(tk.END, "Evidence rating: ", "bold")
                    annot_text.insert(tk.END, f"{action.get('Evidence_label', '')}\n")
                if has_evidence_thoughts:
                    annot_text.insert(tk.END, "Thoughts: ", "bold")
                    annot_text.insert(tk.END, f"{action.get('Evidence_thoughts', '')}\n")

            annot_text.config(state=tk.DISABLED)
            
            # Get chunk content
            chunk_num = action.get("Chunk", "Unknown")
            chunk_content = chunks.get(str(chunk_num), "Chunk content not found")
            
            # Update chunk content with character name highlighting
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            
            # Split character name into parts to handle first name, last name, etc.
            char_parts = character.split()
            
            # Insert text with character name highlighting
            remaining_text = chunk_content
            current_pos = 0
            
            while remaining_text:
                # Find the earliest occurrence of any character name part (case-insensitive)
                earliest_match = None
                earliest_pos = len(remaining_text)
                matched_part = ""
                
                for part in char_parts:
                    if len(part) >= 2:  # Only highlight parts with 2+ characters
                        pos = remaining_text.lower().find(part.lower())
                        if pos != -1 and pos < earliest_pos:
                            earliest_pos = pos
                            earliest_match = pos
                            matched_part = part
                
                if earliest_match is not None:
                    # Insert text before the match
                    text_widget.insert(tk.END, remaining_text[:earliest_pos])
                    
                    # Insert the matched character name part in bold
                    actual_match = remaining_text[earliest_pos:earliest_pos + len(matched_part)]
                    text_widget.insert(tk.END, actual_match, "char_bold")
                    
                    # Update remaining text
                    remaining_text = remaining_text[earliest_pos + len(matched_part):]
                else:
                    # No more matches, insert remaining text
                    text_widget.insert(tk.END, remaining_text)
                    break
            
            text_widget.config(state=tk.DISABLED)
            
        else:
            root.destroy()

    def make_label_handler(label):
        def handler():
            character, action = annotations[idx[0]]
            new_entry = {
                "character": character,
                "annotation": action,
                "human_label": label
            }
            result.append(new_entry)
            
            # Immediately save the new entry to file (append mode)
            with open(outputfile, "a") as f:
                f.write(json.dumps(new_entry) + "\n")
            
            idx[0] += 1
            update_annotation()
        return handler

    # Create enhanced buttons dynamically with black text
    button_colors = [
        ("#dc3545", "#c82333"),  # Red
        ("#fd7e14", "#e8620c"),  # Orange  
        ("#ffc107", "#e0a800"),  # Yellow
        ("#28a745", "#218838"),  # Green
        ("#007bff", "#0056b3"),  # Blue
        ("#6f42c1", "#5a32a3"),  # Purple
        ("#17a2b8", "#138496"),  # Teal
        ("#343a40", "#23272b")   # Dark
    ]
    
    for i, label in enumerate(labels):
        bg_color, hover_color = button_colors[i % len(button_colors)]
        
        btn = tk.Button(
            btn_container, 
            text=f"  {label.title()}  ",
            command=make_label_handler(label), 
            font=("Segoe UI", 11, "bold"),
            bg=bg_color,
            fg='black',  # Changed to black text
            activebackground=hover_color,
            activeforeground='black',  # Changed to black text
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        btn.pack(side=tk.LEFT, padx=8, pady=15)
        
        # Add hover effects
        def on_enter(e, btn=btn, color=hover_color):
            btn.config(bg=color)
        def on_leave(e, btn=btn, color=bg_color):
            btn.config(bg=color)
            
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    # Add keyboard shortcuts
    def on_key(event):
        key = event.keysym
        if key.isdigit() and int(key) <= len(labels):
            idx_key = int(key) - 1
            if 0 <= idx_key < len(labels):
                make_label_handler(labels[idx_key])()

    root.bind('<Key>', on_key)
    root.focus_set()

    # Instructions label
    instructions = tk.Label(
        root, 
        text=f"💡 Use number keys 1-{len(labels)} for quick selection or click buttons above",
        font=("Segoe UI", 9, "italic"),
        fg='#6c757d',
        bg='#f8f9fa',
        pady=5
    )
    instructions.pack()

    update_annotation()
    root.mainloop()

    # Calculate total number of evaluated annotations
    n_total = len(result)

    # Update summary statistics to include all labels
    summary_stats = {
        "n_total": n_total,
        "labels": labels
    }
    
    # Add counts and proportions for each label
    for label in labels:
        n_label = sum(1 for r in result if r["human_label"] == label)
        prop_label = n_label / n_total if n_total > 0 else 0
        
        # Bayesian credible interval for this label
        alpha = 1 + n_label
        beta_param = 1 + n_total - n_label
        lower = beta.ppf(0.025, alpha, beta_param)
        upper = beta.ppf(0.975, alpha, beta_param)
        
        label_key = label.replace(' ', '_')
        summary_stats[f"n_{label_key}"] = n_label
        summary_stats[f"proportion_{label_key}"] = prop_label
        summary_stats[f"95%_CI_lower_{label_key}"] = lower
        summary_stats[f"95%_CI_upper_{label_key}"] = upper

    # Save to summary file
    with open(summaryfile, "w") as f:
        json.dump(summary_stats, f, indent=2)


def annotate(
    chunkfile,
    traits=None,
    target_characters=None,
    outputfile=None,
    model="gpt-4.1-mini",
    book_title='Unknown title'
):
    """Annotate character actions for specific traits and/or characters.
    
    Args:
        chunkfile: Path to text chunks JSON
        traits: Dict of {trait_name: {trait_explanation, positive_examples, negative_examples}}
        target_characters: List of specific character names to annotate
        outputfile: Path to save annotations JSON
        model: LLM model for annotation
        book_title: Book title for context
    
    Returns:
        Dict of {character: [annotation_dicts]}
    """
    with open(chunkfile, "r", encoding="utf-8") as f:
        chunks_dicts = json.load(f)
    chunks = list(chunks_dicts.values())

    # Determine characters to annotate
    if target_characters:
        characters = target_characters
    else:
        characters = []

    all_annotations = defaultdict(list)
    for k, section in enumerate(chunks):
        if k+1 % 50 == 0:
            print(f"Annotating chunk {k+1}/{len(chunks)}")

        # For each chunk, annotate for each character and/or trait
        for char in characters if characters else [None]:
            # If both traits and characters: combine prompts
            if traits and char:
                for trait, info in traits.items():
                    trait_explanation = info['trait_explanation']
                    pos = info.get("positive_examples", info.get("postive_examples", ""))
                    neg = info.get("negative_examples", "")
                    trait_action_examples = []
                    if pos:
                        trait_action_examples.append(f'{{"Character Name": "{char}", "Action": "{pos}", "{trait.capitalize()}": 1}}')
                    if neg:
                        trait_action_examples.append(f'{{"Character Name": "{char}", "Action": "{neg}", "{trait.capitalize()}": -1}}')
                    trait_action_examples_str = ", ".join(trait_action_examples)
                    prompt = (
                        f"<Instructions> You analyze the behavior of the character: {char}. You read excerpts from the book {book_title} and pay close attention to characters' actions, statements, internal behaviors, and prominent non-actions/omissions.\n"
                        f"Your goal is to find positive and negative indicators of the following trait: {trait} for the character {char}.\n"
                        f"Explanation of trait:\n{trait.capitalize()}: {trait_explanation}\n"
                        f"Example output: [{trait_action_examples_str}, ...]\n"
                        f"Be measured in your judgment of valid trait indicators.\n"
                        f"Only return a list of dicts with keys 'Character Name', 'Action', and '{trait.capitalize()}' and only use these scores: -1, 0, 1.</Instructions> \n\n"
                        f"<Metadata>Text source: {book_title}; character: {char}</Metadata>\n"
                        f"<Excerpt>\n\n...{section}...\n\n</Excerpt>"
                    )
                    response = call_model(model, prompt, temperature=0.0)
                    try:
                        dicts = re.findall(r'\{.*?\}', response)
                    except Exception:
                        print("ERROR parsing response:", response)
                        dicts = []
                    try:
                        dicts = [ast.literal_eval(d) for d in dicts]
                    except Exception:
                        dicts = []
                    for d in dicts:
                        for key, value in d.items():
                            if isinstance(value, str) and key.lower() != "character name" and key != "Action":
                                d["Action"] = value
                                del d[key]
                        d["Chunk"] = k+1
                        if "Character Name" in d:
                            name = d.pop("Character Name", None)
                            all_annotations[name].append(d)
            elif traits and not char:
                # Trait annotation only (no specific character)
                for trait, info in traits.items():
                    trait_explanation = info['trait_explanation']
                    pos = info.get("positive_examples", info.get("postive_examples", ""))
                    neg = info.get("negative_examples", "")
                    trait_action_examples = []
                    if pos:
                        trait_action_examples.append(f'{{"Character Name": "John Doe", "Action": "{pos}", "{trait.capitalize()}": 1}}')
                    if neg:
                        trait_action_examples.append(f'{{"Character Name": "Jane Doe", "Action": "{neg}", "{trait.capitalize()}": -1}}')
                    trait_action_examples_str = ", ".join(trait_action_examples)
                    prompt = (
                        f"<Instructions> You analyze the behavior of fiction characters. You read excerpts from books and pay close attention to characters' actions, statements, internal behaviors, and prominent non-actions/omissions.\n"
                        f"Your goal is to find positive and negative indicators of the following trait: {trait}.\n"
                        f"Explanation of trait:\n{trait.capitalize()}: {trait_explanation}\n"
                        f"Example output: [{trait_action_examples_str}, ...]\n"
                        f"Be measured in your judgment of valid trait indicators.\n"
                        f"Only return a list of dicts with keys 'Character Name', 'Action', and '{trait.capitalize()}' and only use these scores: -1, 0, 1.</Instructions> \n\n"
                        f"<Metadata>Text source: {book_title}</Metadata>\n"
                        f"<Excerpt>\n\n...{section}...\n\n</Excerpt>"
                    )
                    response = call_model(model, prompt, temperature=0.0)
                    dicts = re.findall(r'\{.*?\}', response)
                    try:
                        dicts = [ast.literal_eval(d) for d in dicts]
                    except Exception:
                        dicts = []
                    for d in dicts:
                        for key, value in d.items():
                            if isinstance(value, str) and key.lower() != "character name" and key != "Action":
                                d["Action"] = value
                                del d[key]
                        d["Chunk"] = k+1
                        if "Character Name" in d:
                            name = d.pop("Character Name", None)
                            all_annotations[name].append(d)
            elif char and not traits:
                # Character annotation only
                prompt = (
                    f"<Instructions> You analyze fiction characters by closely observing and annotating their actions and thoughts. "
                    f"Your target is a character named: {char}.\n"
                    """Read through the following excerpt and return a comprehensive list of dictionaries {"Action": "*short description*", "Inferred traits": ["trait1", "trait2", ...] }</Instructions>\n"""
                    f"<Metadata>Text source: {book_title}</Metadata>\n"
                    f"<Excerpt>\n\n...{section}...\n\n</Excerpt>"
                    f"<Reminder> Return only the list of dictionaries for behaviors of {char} with keys Action and Inferred traits.</Reminder>\n\n"
                )
                response_text = call_model(model, prompt, temperature=0)
                dicts = re.findall(r'\{.*?\}', response_text.replace('\n', ' '), re.DOTALL)
                try:
                    dicts = [ast.literal_eval(d) for d in dicts]
                except Exception:
                    dicts = []
                for d in dicts:
                    d["Chunk"] = k+1
                # Reformat: {"Action": ..., "Inferred traits": [...]} -> {"Action": ..., trait: 1, ...}
                reformatted_dicts = []
                for d in dicts:         #if there is a string value and the key.lower() is not character name, make the key "Action" (necessary because sometimes LLM writes 'non-action' etc)
                    for key, value in d.items():
                        if isinstance(value, str) and key.lower() != "character name" and key != "Action":
                            d["Action"] = value
                            del d[key]
                    action = d.get("Action", "")
                    traits_found = d.get("Inferred traits", [])
                    for trait in traits_found:
                        reformatted_dicts.append({"Action": action, trait: 1, "Chunk": d.get("Chunk")})
                for d in reformatted_dicts:
                    all_annotations[char].append(d)

            else: #neither traits nor characters
                prompt = (
                        f"<Instructions> You analyze the behavior of fiction characters. You read excerpts from books and pay close attention to characters' actions, statements, internal behaviors, and prominent non-actions/omissions.\n"
                        """Read through the following excerpt and return a comprehensive list of dictionaries {"Character Name": "John Doe", "Action": "*short description*", "Inferred traits": ["trait1", "trait2", ...] }</Instructions>\n"""
                        f"<Metadata>Text source: {book_title}</Metadata>\n"
                        f"<Excerpt>\n\n...{section}...\n\n</Excerpt>"
                        f"<Reminder> Return only the list of dictionaries with keys Character Name, Action, and Inferred traits.</Reminder>\n\n"
                    )
                response = call_model(model, prompt, temperature=0.0)
                dicts = re.findall(r'\{.*?\}', response, re.DOTALL)
                try:
                    dicts = [ast.literal_eval(d) for d in dicts]
                except Exception:
                    dicts = []

                reformatted_dicts = []
                for d in dicts:
                    #if there is a string value and the key.lower() is not character name, make the key "Action" (necessary because sometimes LLM writes 'non-action' etc)
                    for key, value in d.items():
                        if isinstance(value, str) and key.lower() != "character name" and key != "Action":
                            d["Action"] = value
                            del d[key]
                    action = d.get("Action", "")
                    traits_found = d.get("Inferred traits", [])
                    for trait in traits_found:
                        reformatted_dicts.append({"Character Name": d.get("Character Name"), "Action": action, trait: 1, "Chunk": k+1})
                for d in reformatted_dicts:
                    char = d.pop("Character Name", None)
                    all_annotations[char].append(d)

    #print total number of annotations
    total_annotations = sum(len(v) for v in all_annotations.values())
    print(f"Total annotations collected: {total_annotations}")
    with open(outputfile, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)

