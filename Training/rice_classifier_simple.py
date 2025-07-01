import numpy as np
import logging
from typing import Dict, List, Optional

class RiceClassifier:
    """Simple rice grain classification demo without TensorFlow dependency"""
    
    def __init__(self):
        """Initialize the classifier"""
        self.model = "demo_mode"
        self.class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        
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
        
        logging.info("✅ Rice classifier (demo mode) initialized successfully")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def predict(self, image: np.ndarray) -> Optional[Dict]:
        """Make a basic prediction"""
        return self.enhanced_predict(image)
    
    def enhanced_predict(self, image: np.ndarray) -> Optional[Dict]:
        """Enhanced prediction with realistic demo results"""
        try:
            # Generate realistic demo predictions
            predicted_class = np.random.choice(self.class_names)
            confidence = np.random.uniform(75, 95)
            
            # Create realistic prediction probabilities
            all_predictions = []
            remaining_prob = 100.0 - confidence
            other_classes = [cls for cls in self.class_names if cls != predicted_class]
            
            # Distribute remaining probability among other classes
            for i, cls in enumerate(other_classes):
                if i == len(other_classes) - 1:
                    prob = remaining_prob
                else:
                    prob = np.random.uniform(1, remaining_prob * 0.4)
                    remaining_prob -= prob
                
                all_predictions.append({
                    'class': cls,
                    'probability': round(prob, 1)
                })
            
            # Add the predicted class
            all_predictions.append({
                'class': predicted_class,
                'probability': round(confidence, 1)
            })
            
            # Sort by probability (highest first)
            all_predictions.sort(key=lambda x: x['probability'], reverse=True)
            
            # Get recommendations for predicted class
            recommendations = self.recommendations.get(predicted_class, {})
            
            logging.info(f"Demo prediction: {predicted_class} with {confidence:.1f}% confidence")
            
            return {
                'predicted_class': predicted_class,
                'confidence': round(confidence, 1),
                'all_predictions': all_predictions,
                'recommendations': recommendations
            }
            
        except Exception as e:
            import traceback
            logging.error(f"Error making prediction: {str(e)} (type: {type(e)})\n{traceback.format_exc()}")
            return None

    def get_class_info(self, class_name: str) -> Dict:
        """Get information about a specific rice class"""
        return self.recommendations.get(class_name, {})
