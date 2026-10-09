from pv_triage.agents.phi import leaks, redact


def test_redacts_each_phi_type():
    text = (
        "Patient name: Jane Doe\n"
        "Reporter name: Alan Smith\n"
        "Phone: (555) 201-4417\n"
        "Email: jane.doe@example.com\n"
        "SSN: 123-45-6789\n"
        "MRN: 00482913\n"
        "DOB: 03/12/1959\n"
        "Seen by Dr. Helen Ortiz at 412 Maple Street."
    )
    out, counts, removed = redact(text)
    assert counts == {"EMAIL": 1, "SSN": 1, "PHONE": 1, "MRN": 1, "DOB": 1,
                      "ADDRESS": 1, "NAME": 3}
    for value in ["Jane Doe", "Alan Smith", "201-4417", "jane.doe@example.com",
                  "123-45-6789", "00482913", "03/12/1959", "Helen Ortiz", "Maple Street"]:
        assert value not in out


def test_keeps_clinical_fields_and_line_structure():
    text = "Reporter name: Priya Natarajan\nEmail: p@example.com\nAge: 52\nSex: Male"
    out, _, _ = redact(text)
    assert out.splitlines() == ["Reporter name: [NAME]", "Email: [EMAIL]", "Age: 52", "Sex: Male"]


def test_leak_check_finds_identifier_without_its_label():
    _, _, removed = redact("MRN: 00482913")
    assert leaks("the record 00482913 shows", removed) == ["00482913"]
    assert leaks("nothing identifying here", removed) == []
