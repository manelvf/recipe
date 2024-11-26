"""Main server for API entrypoint"""
import uvicorn
from fastapi import FastAPI

from .recipe import Recipe, load_recipes


recipes = load_recipes()
recipe = Recipe(recipes)

app = FastAPI()


@app.get("/recommend_recipe")
async def root(ingredients: str | None = "", image: int | None = None):
    if image:
        image_ingredients = recipe.get_image_ingredients(image)
    ingredients = f"{ingredients}, {image_ingredients}"

    return recipe.get_recipe(ingredients)


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("my_package.main:app", host="0.0.0.0", port=8000, reload=True)
