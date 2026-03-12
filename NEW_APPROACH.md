# New Hierarchical Prediction & Guidance Model

## Problem Statement

You're building a system that:
1. **Learns from experienced experimentalists** (training on FineBio)
2. **Watches newbies perform experiments** (real-time inference)
3. **Predicts their next step** (future action prediction)
4. **Detects errors and guides them** (when they deviate from expected protocol)

FineBio dataset has inherent problems that will exist in real environments:
- Variable ordering of atomic operations (flexible within strict step sequences)
- Hand-object interaction ambiguity
- Multi-granularity temporal understanding (atomic ops → steps → protocols)
- Occlusion and multi-view synchronization issues

## Why Previous Approaches Failed

1. **Simple GRU with object counts**: Too weak - just counts don't capture spatial relationships
2. **2D-only features**: Missing 3D spatial understanding for hand-object interactions
3. **No protocol awareness**: Doesn't respect step ordering constraints
4. **No error detection**: Can't identify when user goes wrong
5. **No guidance generation**: Can't provide actionable feedback

## New Architecture: Hierarchical Prediction Model

### Core Components

#### 1. **Object Relationship Graph (GNN)**
- **Purpose**: Model spatial relationships between objects
- **Nodes**: Object classes (35 lab objects)
- **Edges**: Spatial proximity, hand-object interactions, object-object relationships
- **Why it works**: Captures which objects are near each other, which hand is manipulating which object

```python
ObjectRelationshipGraph:
  - Node embeddings: object class → feature vector
  - Edge features: [distance_2d, distance_3d, IoU, hand_interaction]
  - Graph attention: learns which objects interact
```

#### 2. **Protocol-Aware Transformer**
- **Purpose**: Temporal sequence modeling that respects protocol structure
- **Key innovation**: Step embeddings enforce strict step ordering, while attention allows flexible atomic operations
- **Why it works**: 
  - Steps must follow protocol order (strict constraint)
  - Atomic operations within steps can be flexible (attention learns patterns)
  - Positional encoding captures temporal context

```python
ProtocolAwareTransformer:
  - Step embeddings: learnable protocol structure
  - Positional encoding: temporal context
  - Multi-head attention: flexible atomic operation patterns
  - Three prediction heads:
    * Current step prediction
    * Atomic operation prediction
    * Next-step prediction (future)
```

#### 3. **Error Detection Head**
- **Purpose**: Detect when user deviates from expected sequence
- **Method**: Contrastive learning - compare predicted vs actual
- **Why it works**: Learns to distinguish correct sequences from incorrect ones

```python
ErrorDetectionHead:
  - Binary classification: error or not
  - Error type classification: wrong step, wrong order, missing object, etc.
  - Uses contrastive learning: correct vs incorrect sequences
```

#### 4. **Guidance Generation**
- **Purpose**: Provide actionable feedback when errors detected
- **Input**: Current state + error information
- **Output**: Next step or atomic operation to perform

```python
GuidanceHead:
  - Takes: [current_state, error_info]
  - Outputs: next step or atomic operation recommendation
```

### Multi-Task Learning Objectives

The model simultaneously learns:
1. **Step prediction** (coarse-grained): What step is happening now?
2. **Atomic operation prediction** (fine-grained): What atomic op is happening now?
3. **Next-step prediction** (future): What step should come next?
4. **Error detection**: Is the user following the correct sequence?

Loss function:
```
total_loss = 1.0 * step_loss + 
             0.5 * atomic_loss + 
             1.5 * next_step_loss +  # Emphasize future prediction
             0.3 * error_loss
```

## How It Addresses FineBio Challenges

### 1. Variable Atomic Operation Ordering
- **Problem**: Atomic ops can occur in any order within a step
- **Solution**: Attention mechanism in transformer learns flexible patterns
- **Result**: Model adapts to different execution orders while respecting step constraints

### 2. Hand-Object Interaction Recognition
- **Problem**: Hard to tell which object is being manipulated
- **Solution**: Graph neural network models spatial relationships
- **Integration**: 3D reconstruction provides true 3D hand-object distances
- **Result**: Better understanding of manipulation states

### 3. Multi-Granularity Temporal Understanding
- **Problem**: Need to understand actions at multiple scales
- **Solution**: Hierarchical prediction (atomic ops + steps + next steps)
- **Result**: Model operates at all three levels simultaneously

### 4. Error Detection & Guidance
- **Problem**: Need to detect when user goes wrong and guide them
- **Solution**: Contrastive learning + guidance generation
- **Result**: Real-time error detection with actionable feedback

## Integration with Existing Code

### From `yolo26.ipynb`:
- ✅ Multi-view detection (FPV + TPV fusion)
- ✅ Rich feature extraction (object counts + temporal deltas)
- ✅ FineBio-trained YOLO-26 weights

### From `threeD_reconstruction.ipynb`:
- ✅ 3D triangulation of objects and hands
- ✅ Hand-object distance computation
- ✅ ROI detection (manipulated/affected objects)

### New Components:
- ✅ Graph neural network for object relationships
- ✅ Protocol-aware transformer
- ✅ Error detection head
- ✅ Guidance generation

## Training Strategy

### Phase 1: Pre-training on FineBio
1. Train on experienced users' sequences (correct executions)
2. Learn protocol structure and step ordering
3. Learn atomic operation patterns

### Phase 2: Contrastive Learning
1. Use mistake trials (if available) or synthetic errors
2. Train error detection: distinguish correct vs incorrect
3. Learn error types: wrong step, wrong order, missing object

### Phase 3: Fine-tuning for Guidance
1. Train guidance head on error scenarios
2. Learn to generate actionable feedback
3. Optimize for real-time inference

## Real-Time Inference Pipeline

```
1. Video frames (FPV + TPV) 
   ↓
2. YOLO-26 detection (objects + hands)
   ↓
3. 3D triangulation (if available)
   ↓
4. Feature extraction:
   - Object counts
   - Graph features (spatial relationships)
   - 3D spatial features (hand-object distances)
   ↓
5. Hierarchical Prediction Model:
   - Current step prediction
   - Atomic operation prediction
   - Next-step prediction
   - Error detection
   ↓
6. Guidance generation (if error detected)
   ↓
7. Output: Next step + guidance message
```

## Expected Improvements

1. **Better step prediction**: Protocol-aware structure improves accuracy
2. **Better hand-object understanding**: Graph features capture spatial relationships
3. **Future prediction**: Next-step prediction enables proactive guidance
4. **Error detection**: Can identify when user deviates
5. **Actionable guidance**: Provides specific feedback on what to do next

## Next Steps

1. **Implement and train** the hierarchical model
2. **Integrate 3D features** from reconstruction pipeline
3. **Collect error data** (mistake trials or synthetic errors)
4. **Fine-tune for guidance** generation
5. **Deploy for real-time** inference on newbie experiments

## Key Files

- `hierarchical_prediction_model.py`: Core model architecture
- `train_hierarchical_model.ipynb`: Training notebook
- `threeD_reconstruction.ipynb`: 3D features (integrate with model)
- `yolo26.ipynb`: Object detection and feature extraction
