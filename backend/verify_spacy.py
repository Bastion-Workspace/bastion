#!/usr/bin/env python3
"""
SpaCy verification script for Docker container
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_spacy():
    """Verify spaCy installation and model availability"""
    try:
        logger.info("🔍 Checking spaCy installation...")
        import spacy
        logger.info(f"✅ spaCy version: {spacy.__version__}")
        
        # Check for en_core_web_lg model specifically
        import spacy.util
        model_name = "en_core_web_lg"
        
        if spacy.util.is_package(model_name):
            logger.info(f"✅ Model available: {model_name}")
        else:
            logger.error(f"❌ Model not available: {model_name}")
            logger.info("🔧 Installing en_core_web_lg...")
            import subprocess
            result = subprocess.run([sys.executable, "-m", "spacy", "download", model_name], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ {model_name} installed successfully")
            else:
                logger.error(f"❌ Failed to install model: {result.stderr}")
                return False
        
        # Test model loading and processing
        try:
            logger.info(f"🔍 Testing model: {model_name}")
            nlp = spacy.load(model_name)
            
            # Test basic processing
            test_text = "Apple Inc. is headquartered in Cupertino, California. Tim Cook is the CEO."
            doc = nlp(test_text)
            
            logger.info(f"✅ Model {model_name} loaded successfully")
            logger.info(f"✅ Processing test: {len(doc.ents)} entities found")
            
            # List found entities
            for ent in doc.ents:
                logger.info(f"   - {ent.text} ({ent.label_})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to test model {model_name}: {e}")
            return False
        
    except ImportError as e:
        logger.error(f"❌ spaCy not installed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during spaCy verification: {e}")
        return False

if __name__ == "__main__":
    success = verify_spacy()
    sys.exit(0 if success else 1) 