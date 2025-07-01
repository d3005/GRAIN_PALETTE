import os
import tensorflow as tf
import numpy as np
import logging
from typing import Dict, List, Optional

class RiceClassifier:
    """Rice grain classification model using MobileNetV2 transfer learning"""
    
    def __init__(self):
        """Initialize the classifier"""
        self.model = None
        self.class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        self.class_labels = {
            'arborio': 0,
            'basmati': 1,
            'ipsala': 2,
            'jasmine': 3,
            'karacadag': 4
        }
        
        # Agricultural recommendations for each rice type
        self.recommendations = {
            'Arborio': {
                'description': 'Short-grain rice ideal for risotto and Mediterranean dishes.',
                'cultivation': 'Requires consistent moisture and warm temperatures. Best grown in flooded fields.',
                'water_needs': 'High water requirement - maintain flooded conditions during growing season.',
                'fertilizer': 'Use balanced NPK fertilizer. Apply nitrogen in split doses.',
                'harvest_time': '120-150 days from planting'
            },
            'Basmati': {
                'description': 'Long-grain aromatic rice prized for its fragrance and fluffy texture.',
                'cultivation': 'Grows best in well-drained, fertile soil with good organic matter.',
                'water_needs': 'Moderate to high water requirement. Avoid waterlogging during grain filling.',
                'fertilizer': 'Organic matter-rich soil preferred. Use phosphorus-rich fertilizer.',
                'harvest_time': '120-140 days from planting'
            },
            'Ipsala': {
                'description': 'Turkish rice variety known for its cooking qualities and grain structure.',
                'cultivation': 'Adapted to Mediterranean climate conditions. Requires careful water management.',
                'water_needs': 'Moderate water requirement with good drainage.',
                'fertilizer': 'Balanced fertilization with emphasis on potassium for grain quality.',
                'harvest_time': '130-150 days from planting'
            },
            'Jasmine': {
                'description': 'Fragrant long-grain rice popular in Asian cuisine.',
                'cultivation': 'Thrives in tropical and subtropical conditions with high humidity.',
                'water_needs': 'High water requirement during vegetative growth, reduce before harvest.',
                'fertilizer': 'Nitrogen-rich fertilizer in early stages, reduce during flowering.',
                'harvest_time': '110-130 days from planting'
            },
            'Karacadag': {
                'description': 'Traditional Turkish rice variety with excellent cooking properties.',
                'cultivation': 'Hardy variety suitable for various soil types and climate conditions.',
                'water_needs': 'Moderate water requirement with tolerance to slight water stress.',
                'fertilizer': 'Responds well to organic fertilizers and balanced NPK application.',
                'harvest_time': '125-145 days from planting'
            }
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load rice_classifier.h5 model with multiple fallback strategies"""
        try:
            import keras
            import tensorflow as tf
            logging.info("Loading rice_classifier.h5 model only...")
            h5_fallback_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rice_classifier.h5')
            
            if os.path.exists(h5_fallback_path):
                # Try loading without custom objects first
                try:
                    self.model = tf.keras.models.load_model(h5_fallback_path, compile=False)
                    logging.info(f"Successfully loaded H5 model from {h5_fallback_path}")
                except Exception as e1:
                    logging.warning(f"Failed to load without custom objects: {str(e1)}")
                    
                    # Try with custom object scope for common preprocessing issues
                    try:
                        # Define TrueDivide as a custom layer class
                        class TrueDivide(tf.keras.layers.Layer):
                            def __init__(self, divisor=127.5, **kwargs):
                                super(TrueDivide, self).__init__(**kwargs)
                                self.divisor = divisor
                            
                            def call(self, inputs):
                                return tf.math.truediv(inputs, self.divisor)
                            
                            def get_config(self):
                                config = super(TrueDivide, self).get_config()
                                config.update({'divisor': self.divisor})
                                return config
                        
                        with keras.utils.custom_object_scope({
                            'TrueDivide': TrueDivide,
                            'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
                        }):
                            self.model = tf.keras.models.load_model(h5_fallback_path, compile=False)
                        logging.info(f"Successfully loaded H5 model with custom objects from {h5_fallback_path}")
                    except Exception as e2:
                        logging.warning(f"Failed to load with custom objects: {str(e2)}")
                        
                        # Try loading SavedModel format as fallback using TFSMLayer
                        try:
                            savedmodel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rice_classifier_savedmodel')
                            if os.path.exists(savedmodel_path):
                                # Use TFSMLayer for Keras 3 compatibility
                                self.model = tf.keras.layers.TFSMLayer(savedmodel_path, call_endpoint='serving_default')
                                logging.info(f"Successfully loaded SavedModel using TFSMLayer from {savedmodel_path}")
                            else:
                                logging.error("SavedModel format not found either")
                                self.model = None
                        except Exception as e3:
                            logging.error(f"Failed to load SavedModel: {str(e3)}")
                            self.model = None
                
                if self.model is not None:
                    # Only log shapes for regular Keras models, not TFSMLayer
                    if hasattr(self.model, 'input_shape') and hasattr(self.model, 'output_shape'):
                        logging.info(f"Model input shape: {self.model.input_shape}")
                        logging.info(f"Model output shape: {self.model.output_shape}")
                    else:
                        logging.info("Model loaded successfully (TFSMLayer format)")
            else:
                logging.error(f"rice_classifier.h5 not found at {h5_fallback_path}")
                self.model = None
                
        except Exception as e:
            logging.error(f"Error loading rice_classifier.h5: {str(e)}")
            self.model = None

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def predict(self, image: np.ndarray) -> Optional[Dict]:
        """
        Make prediction on preprocessed image
        
        Args:
            image: Preprocessed image array of shape (1, 224, 224, 3)
            
        Returns:
            Dictionary containing prediction results or None if error
        """
        if not self.is_loaded():
            logging.error("Model not loaded")
            return None
        
        try:
            # Handle TFSMLayer differently than regular Keras models
            if hasattr(self.model, '_saved_model_path'):
                # This is a TFSMLayer - call it with the inputs
                predictions = self.model(image)
                # TFSMLayer returns a dictionary, extract the output
                if isinstance(predictions, dict):
                    # Get the first (and likely only) output
                    predictions = list(predictions.values())[0]
            else:
                # Regular Keras model
                predictions = self.model(image, training=False)

            # If predictions is still a dict, extract the first value
            if isinstance(predictions, dict):
                predictions = list(predictions.values())[0]

            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
            
            # Apply softmax if the outputs are logits (raw outputs without softmax)
            # Check if values don't sum to ~1.0 (indicating they need softmax)
            if abs(np.sum(predictions[0]) - 1.0) > 0.1:
                predictions = tf.nn.softmax(predictions).numpy()
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx]) * 100
            predicted_class = self.class_names[predicted_class_idx]
            
            # Get all class probabilities
            all_predictions = []
            for i, class_name in enumerate(self.class_names):
                prob = float(predictions[0][i]) * 100
                all_predictions.append({
                    'class': class_name,
                    'probability': round(prob, 2)
                })
            
            # Sort by probability (highest first)
            all_predictions.sort(key=lambda x: x['probability'], reverse=True)
            
            # Get recommendations for predicted class
            recommendations = self.recommendations.get(predicted_class, {})
            
            logging.info(f"Prediction: {predicted_class} with {confidence:.2f}% confidence")
            logging.debug(f"All predictions: {[f'{p['class']}: {p['probability']:.2f}%' for p in all_predictions]}")
            
            return {
                'predicted_class': predicted_class,
                'confidence': round(confidence, 2),
                'all_predictions': all_predictions,
                'recommendations': recommendations
            }
            
        except Exception as e:
            import traceback
            logging.error(f"Error making prediction: {str(e)} (type: {type(e)})\n{traceback.format_exc()}")
            return None
    
    def enhanced_predict(self, image: np.ndarray) -> Optional[Dict]:
        """
        Enhanced prediction with post-processing heuristics to improve accuracy
        
        Args:
            image: Preprocessed image array of shape (1, 224, 224, 3)
            
        Returns:
            Dictionary containing enhanced prediction results or None if error
        """
        # Get base prediction
        result = self.predict(image)
        if not result:
            return None
        
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        all_predictions = result['all_predictions']
        
        # Create a mapping for easier access
        prob_map = {pred['class']: pred['probability'] for pred in all_predictions}
        
        # Apply refined heuristics based on observed confusion patterns
        
        # Rule 1: If Karacadag is predicted with low confidence and Arborio is close, prefer Arborio
        if predicted_class == 'Karacadag':
            arborio_prob = prob_map.get('Arborio', 0)
            confidence_diff = confidence - arborio_prob
            
            # Only change if Arborio is very close (within 7%) and confidence is low
            if confidence_diff < 7 and confidence < 32 and arborio_prob > 20:
                logging.info(f"Enhanced prediction: Switching from Karacadag to Arborio (confidence diff: {confidence_diff:.2f}%)")
                # Swap to Arborio
                for pred in all_predictions:
                    if pred['class'] == 'Arborio':
                        pred['probability'] = confidence + 3
                    elif pred['class'] == 'Karacadag':
                        pred['probability'] = confidence - 3
                
                all_predictions.sort(key=lambda x: x['probability'], reverse=True)
                result['predicted_class'] = 'Arborio'
                result['confidence'] = all_predictions[0]['probability']
                result['all_predictions'] = all_predictions
        
        # Rule 2: If Basmati is predicted with low confidence and Jasmine is close, prefer Jasmine
        elif predicted_class == 'Basmati':
            jasmine_prob = prob_map.get('Jasmine', 0)
            confidence_diff = confidence - jasmine_prob
            
            # Only change if Jasmine is very close (within 5%) and confidence is very low
            if confidence_diff < 5 and confidence < 25 and jasmine_prob > 18:
                logging.info(f"Enhanced prediction: Switching from Basmati to Jasmine (confidence diff: {confidence_diff:.2f}%)")
                # Swap to Jasmine
                for pred in all_predictions:
                    if pred['class'] == 'Jasmine':
                        pred['probability'] = confidence + 2
                    elif pred['class'] == 'Basmati':
                        pred['probability'] = confidence - 2
                
                all_predictions.sort(key=lambda x: x['probability'], reverse=True)
                result['predicted_class'] = 'Jasmine'
                result['confidence'] = all_predictions[0]['probability']
                result['all_predictions'] = all_predictions
        
        # Get recommendations for final predicted class
        result['recommendations'] = self.recommendations.get(result['predicted_class'], {})
        
        return result

    def get_class_info(self, class_name: str) -> Dict:
        """Get information about a specific rice class"""
        return self.recommendations.get(class_name, {})
