import textwrap
from typing import Optional

from ruamel.yaml.comments import CommentedMap


def _add_comments(
    loaded_yaml: CommentedMap,
    schema: dict,
    indent: int = 0,
    definitions: Optional[dict] = None,
):
    """Add comments above each YAML field using the pydantic model schema."""
    # Definitions are in schemas that contain nested pydantic Models
    if definitions is None:
        definitions = schema.get("definitions")

    for key, val in schema["properties"].items():
        reference = ""
        # Get sub-schema if it exists
        if "$ref" in val.keys():
            # At top level, example is 'outputs': {'$ref': '#/definitions/Outputs'}
            reference = val["$ref"]
        elif "allOf" in val.keys():
            # within 'definitions', it looks like
            #  'allOf': [{'$ref': '#/definitions/HalfWindow'}]
            reference = val["allOf"][0]["$ref"]

        ref_key = reference.split("/")[-1]
        if ref_key:  # The current property is a reference to something else
            if "enum" in definitions[ref_key]:  # type: ignore
                # This is just an Enum, not a sub schema.
                # Overwrite the value with the referenced value
                val = definitions[ref_key]  # type: ignore
            else:
                # The reference is a sub schema, so we need to recurse
                sub_schema = definitions[ref_key]  # type: ignore
                # Get the sub-model
                sub_loaded_yaml = loaded_yaml[key]
                # recurse on the sub-model
                _add_comments(
                    sub_loaded_yaml,
                    sub_schema,
                    indent=indent + 2,
                    definitions=definitions,
                )
                continue

        # add each description along with the type information
        desc = "\n".join(
            textwrap.wrap(f"{val['description']}.", width=90, subsequent_indent="  ")
        )
        type_str = f"\n  Type: {val['type']}."
        choices = f"\n  Options: {val['enum']}." if "enum" in val.keys() else ""

        # Combine the description/type/choices as the YAML comment
        comment = f"{desc}{type_str}{choices}"
        comment = comment.replace("..", ".")  # Remove double periods

        # Prepend the required label for fields that are required
        is_required = key in schema.get("required", [])
        if is_required:
            comment = "REQUIRED: " + comment

        # This method comes from here
        # https://yaml.readthedocs.io/en/latest/detail.html#round-trip-including-comments
        loaded_yaml.yaml_set_comment_before_after_key(key, comment, indent=indent)
