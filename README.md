# chatbot-socrates

## Dev setup

    # Activate python environment
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt

    # Run the chatbot
    python chatbot.py

## Model training

    # Be warned, this can take ~13 hours if running on your laptop
    python train.py

You can test your resulting model by setting `model_name` in chatbot.py

    model_name = "./fine_tuned_model/"
