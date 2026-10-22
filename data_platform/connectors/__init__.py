"""
Extract + load: talk to a source and save what it sends into Bronze, unchanged. No transformation.
In industry this is the job of tools like Airbyte or Fivetran ("EL" in ELT).

Every source module has the same entry point:
    run(inst, ...) -> list[str]     problems found (empty = all good)
"""
