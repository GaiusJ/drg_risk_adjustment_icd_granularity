"""
Datenklassen für die hierarchische ICD-10-Systematik: Kapitel → Gruppe → Code.
Wird für den Aufbau des ICD-Lookup und die Zuordnung von Einzelcodes zu Gruppen verwendet.
"""

from dataclasses import dataclass

@dataclass
class ICDChapter:
    number: int
    name: str
    groups: list

@dataclass
class ICDGroup:
    number: str
    name: str
    codes: list

@dataclass
class ICDCode:
    code: str
    description: str