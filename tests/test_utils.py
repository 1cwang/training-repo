# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ml_pipelines._utils pure functions."""

from unittest.mock import MagicMock, patch

import pytest

from ml_pipelines._utils import convert_struct, get_pipeline_custom_tags


class TestConvertStruct:
    def test_none_returns_empty_dict(self):
        assert convert_struct(None) == {}

    def test_empty_string_returns_empty_dict(self):
        assert convert_struct("") == {}

    def test_dict_string_is_parsed(self):
        assert convert_struct("{'key': 'value'}") == {"key": "value"}

    def test_nested_dict_is_parsed(self):
        result = convert_struct("{'a': {'b': 1}}")
        assert result == {"a": {"b": 1}}

    def test_list_is_parsed(self):
        result = convert_struct("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_invalid_struct_raises(self):
        with pytest.raises(Exception):
            convert_struct("not valid python")


class TestGetPipelineCustomTags:
    def test_returns_original_tags_when_module_missing_function(self):
        """Module that has no get_pipeline_custom_tags method should return original tags."""
        tags = {"Key": "project", "Value": "test"}
        # Use a real module that exists but won't have get_pipeline_custom_tags
        result = get_pipeline_custom_tags("ml_pipelines._utils", None, tags)
        assert result == tags

    def test_returns_original_tags_on_import_error(self):
        """ImportError during tag retrieval should be swallowed and return original tags."""
        tags = [{"Key": "env", "Value": "dev"}]
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = get_pipeline_custom_tags("ml_pipelines.nonexistent", None, tags)
        assert result == tags

    def test_returns_original_tags_for_non_ml_pipelines_namespace(self):
        """Module names outside ml_pipelines.* namespace return original tags."""
        tags = [{"Key": "env", "Value": "dev"}]
        result = get_pipeline_custom_tags("nonexistent.module", None, tags)
        assert result == tags

    def test_calls_module_get_pipeline_custom_tags_when_present(self):
        """When the module exports get_pipeline_custom_tags, its return value is used."""
        expected = [{"Key": "project", "Value": "myproj"}]
        fake_module = MagicMock()
        fake_module.get_pipeline_custom_tags.return_value = expected

        with patch("builtins.__import__", return_value=fake_module):
            result = get_pipeline_custom_tags(
                "ml_pipelines.some.pipeline",
                "{'region': 'us-east-1', 'sagemaker_project_arn': 'arn:aws:sagemaker:us-east-1:123:project/p'}",
                {},
            )

        assert result == expected
