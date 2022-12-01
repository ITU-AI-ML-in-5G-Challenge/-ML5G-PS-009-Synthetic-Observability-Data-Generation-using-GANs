import numpy as np
import pandas as pd
import math
import os

import pickle
import pkg_resources
from packaging import version
from collections import namedtuple
import copy
from copy import deepcopy
import torch
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional

import rdt
from rdt.transformers import BayesGMMTransformer, OneHotEncodingTransformer


FIXED_RNG_SEED = 730
TMP_FILE_NAME = '.sample.csv.temp'
DISABLE_TMP_FILE = 'disable'

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'])


def get_package_versions(model=None):

    versions = {}
    try:
        versions['sdv'] = pkg_resources.get_distribution('sdv').version
        versions['rdt'] = pkg_resources.get_distribution('rdt').version
    except pkg_resources.ResolutionError:
        pass

    if model is not None:
        if not isinstance(model, type):
            model = model.__class__

        model_name = model.__module__ + model.__name__

        for lib in ['copulas', 'ctgan', 'deepecho']:
            if lib in model_name or ('hma' in model_name and lib == 'copulas'):
                try:
                    versions[lib] = pkg_resources.get_distribution(lib).version
                except pkg_resources.ResolutionError:
                    pass

    return versions



class DataTransformer(object):

    def __init__(self, max_clusters=10, weight_threshold=0.005):
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    def _fit_continuous(self, data):
        column_name = data.columns[0]
        gm = BayesGMMTransformer()
        gm.fit(data, [column_name])
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components)

    def _fit_discrete(self, data):
        column_name = data.columns[0]
        ohe = OneHotEncodingTransformer()
        ohe.fit(data, [column_name])
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name, column_type='discrete', transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories)

    def fit(self, raw_data, discrete_columns=()):
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            # work around for RDT issue #328 Fitting with numerical column names fails
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []
        for column_name in raw_data.columns:
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        data[column_name] = data[column_name].to_numpy().flatten()
        gm = column_transform_info.transform
        transformed = gm.transform(data, [column_name])

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        column_data_list = []
        for column_transform_info in self._column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data))

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_types()))
        data.iloc[:, 1] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data, [column_transform_info.column_name])

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_types()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def inverse_transform(self, data, sigmas=None):
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st)
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            'discrete_column_id': discrete_counter,
            'column_id': column_id,
            'value_id': np.argmax(one_hot)
        }

    

    
def random_state(function):
    """Set the random state before calling the function.

    Args:
        function (Callable):
            The function to wrap around.
    """
    def wrapper(self, *args, **kwargs):
        if self.random_states is None:
            return function(self, *args, **kwargs)

        else:
            with set_random_states(self.random_states, self.set_random_state):
                return function(self, *args, **kwargs)

    return wrapper



class BaseSynthesizer:

    random_states = None

    def save(self, path):
        """Save the model in the passed `path`."""
        device_backup = self._device
        self.set_device(torch.device('cpu'))
        torch.save(self, path)
        self.set_device(device_backup)

    @classmethod
    def load(cls, path):
        """Load the model stored in the passed `path`."""
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = torch.load(path)
        model.set_device(device)
        return model

    def set_random_state(self, random_state):
        """Set the random state.

        Args:
            random_state (int, tuple, or None):
                Either a tuple containing the (numpy.random.RandomState, torch.Generator)
                or an int representing the random seed to use for both random states.
        """
        if random_state is None:
            self.random_states = random_state
        elif isinstance(random_state, int):
            self.random_states = (
                np.random.RandomState(seed=random_state),
                torch.Generator().manual_seed(random_state),
            )
        elif (
            isinstance(random_state, tuple) and
            isinstance(random_state[0], np.random.RandomState) and
            isinstance(random_state[1], torch.Generator)
        ):
            self.random_states = random_state
        else:
            raise TypeError(
                f'`random_state` {random_state} expected to be an int or a tuple of '
                '(`np.random.RandomState`, `torch.Generator`)')



class Residual(Module):
    """Residual layer for the CTGANSynthesizer."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)
    
    
class Generator(Module):
    """Generator for the CTGANSynthesizer."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data
    
    
class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""

    def __init__(self, data, output_info, log_frequency):
        self._data = data

        def is_discrete_column(column_info):
            return (len(column_info) == 1
                    and column_info[0].activation_fn == 'softmax')

        n_discrete_columns = sum(
            [1 for column_info in output_info if is_discrete_column(column_info)])

        self._discrete_column_matrix_st = np.zeros(
            n_discrete_columns, dtype='int32')

        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sample conditional vector
        max_category = max([
            column_info[0].dim
            for column_info in output_info
            if is_discrete_column(column_info)
        ], default=0)

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum([
            column_info[0].dim
            for column_info in output_info
            if is_discrete_column(column_info)
        ])

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, :span_info.dim] = category_prob
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

    def _random_choice_prob_index(self, discrete_column_id):
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(self, batch):
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(
            np.arange(self._n_discrete_columns), batch)

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        mask[np.arange(batch), discrete_column_id] = 1
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = (self._discrete_column_cond_st[discrete_column_id] + category_id_in_col)
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col

    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None

        cond = np.zeros((batch, self._n_categories), dtype='float32')

        for i in range(batch):
            row_idx = np.random.randint(0, len(self._data))
            col_idx = np.random.randint(0, self._n_discrete_columns)
            matrix_st = self._discrete_column_matrix_st[col_idx]
            matrix_ed = matrix_st + self._discrete_column_n_category[col_idx]
            pick = np.argmax(self._data[row_idx, matrix_st:matrix_ed])
            cond[i, pick + self._discrete_column_cond_st[col_idx]] = 1

        return cond

    def sample_data(self, n, col, opt):
        if col is None:
            idx = np.random.randint(len(self._data), size=n)
            return self._data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        return self._data[idx]

    def dim_cond_vec(self):
        """Return the total number of categories."""
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        """Generate the condition vector."""
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id_ = self._discrete_column_matrix_st[condition_info['discrete_column_id']]
        id_ += condition_info['value_id']
        vec[:, id_] = 1
        return vec


class CTGANSynthesizer(BaseSynthesizer):

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        if version.parse(torch.__version__) < version.parse('1.2.0'):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError('gumbel_softmax returning NaN.')

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)


    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)

class BaseTabularModel:

    _DTYPE_TRANSFORMERS = None
    _metadata = None

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
                 rounding='auto', min_value='auto', max_value='auto'):
        if table_metadata is None:
            self._metadata = Table(
                field_names=field_names,
                primary_key=primary_key,
                field_types=field_types,
                field_transformers=field_transformers,
                anonymize_fields=anonymize_fields,
                constraints=constraints,
                dtype_transformers=self._DTYPE_TRANSFORMERS,
                rounding=rounding,
                min_value=min_value,
                max_value=max_value
            )
            self._metadata_fitted = False
        else:
            table_metadata = deepcopy(table_metadata)
            for arg in (field_names, primary_key, field_types, anonymize_fields, constraints):
                if arg:
                    raise ValueError(
                        'If table_metadata is given {} must be None'.format(arg.__name__))

            if isinstance(table_metadata, dict):
                table_metadata = Table.from_dict(table_metadata)

            table_metadata._dtype_transformers.update(self._DTYPE_TRANSFORMERS)

            self._metadata = table_metadata
            self._metadata_fitted = table_metadata.fitted



    def _sample_rows(self, num_rows, conditions=None, transformed_conditions=None,
                     float_rtol=0.1, previous_rows=None):

        if self._metadata.get_dtypes(ids=False):
            if conditions is None:
                sampled = self._sample(num_rows)
            else:
                try:
                    sampled = self._sample(num_rows, transformed_conditions)
                except NotImplementedError:
                    sampled = self._sample(num_rows)

            sampled = self._metadata.reverse_transform(sampled)

            if previous_rows is not None:

                sampled = pd.concat([previous_rows, sampled], ignore_index=True)

            sampled = self._metadata.filter_valid(sampled)

            if conditions is not None:
                sampled = self._filter_conditions(sampled, conditions, float_rtol)

            num_valid = len(sampled)

            return sampled, num_valid

        else:
            sampled = pd.DataFrame(index=range(num_rows))
            sampled = self._metadata.reverse_transform(sampled)
            return sampled, num_rows

    def _sample_batch(self, num_rows=None, max_tries=100, batch_size_per_try=None,
                      conditions=None, transformed_conditions=None, float_rtol=0.01,
                      progress_bar=None, output_file_path=None):

        if not batch_size_per_try:
            batch_size_per_try = num_rows * 10

        counter = 0
        num_valid = 0
        prev_num_valid = None
        remaining = num_rows
        sampled = pd.DataFrame()

        while num_valid < num_rows:
            if counter >= max_tries:
                break

            prev_num_valid = num_valid
            sampled, num_valid = self._sample_rows(
                batch_size_per_try, conditions, transformed_conditions, float_rtol, sampled,
            )

            num_increase = min(num_valid - prev_num_valid, remaining)
            if num_increase > 0:
                if output_file_path:
                    append_kwargs = {'mode': 'a', 'header': False} if os.path.getsize(
                        output_file_path) > 0 else {}
                    sampled.head(min(len(sampled), num_rows)).tail(num_increase).to_csv(
                        output_file_path,
                        index=False,
                        **append_kwargs,
                    )
                if progress_bar:
                    progress_bar.update(num_increase)

            remaining = num_rows - num_valid
            if remaining > 0:
                LOGGER.info(
                    f'{remaining} valid rows remaining. Resampling {batch_size_per_try} rows')

            counter += 1

        return sampled.head(min(len(sampled), num_rows))



    def _validate_file_path(self, output_file_path):
        """Validate the user-passed output file arg, and create the file."""
        output_path = None
        if output_file_path == DISABLE_TMP_FILE:
            # Temporary way of disabling the output file feature, used by HMA1.
            return output_path

        elif output_file_path:
            output_path = os.path.abspath(output_file_path)
            if os.path.exists(output_path):
                raise AssertionError(f'{output_path} already exists.')

        else:
            if os.path.exists(TMP_FILE_NAME):
                os.remove(TMP_FILE_NAME)

            output_path = TMP_FILE_NAME

        # Create the file.
        with open(output_path, 'w+'):
            pass

        return output_path

    def _randomize_samples(self, randomize_samples):
        if self._model is None:
            return

        if randomize_samples:
            self._set_random_state(None)
        else:
            self._set_random_state(FIXED_RNG_SEED)

    def sample(self, num_rows, randomize_samples=True, batch_size=None, output_file_path=None,
               conditions=None):
        if conditions is not None:
            raise TypeError('This method does not support the conditions parameter. '
                            'Please create `sdv.sampling.Condition` objects and pass them '
                            'into the `sample_conditions` method. '
                            'See User Guide or API for more details.')

        if num_rows is None:
            raise ValueError('You must specify the number of rows to sample (e.g. num_rows=100).')

        if num_rows == 0:
            return pd.DataFrame()

        self._randomize_samples(randomize_samples)

        output_file_path = self._validate_file_path(output_file_path)

        batch_size = min(batch_size, num_rows) if batch_size else num_rows

        sampled = []
        try:
            def _sample_function(progress_bar=None):
                for step in range(math.ceil(num_rows / batch_size)):
                    sampled_rows = self._sample_batch(
                        batch_size,
                        batch_size_per_try=batch_size,
                        progress_bar=progress_bar,
                        output_file_path=output_file_path,
                    )
                    sampled.append(sampled_rows)

                return sampled

            if batch_size == num_rows:
                sampled = _sample_function()
            else:
                sampled = progress_bar_wrapper(_sample_function, num_rows, 'Sampling rows')

        except (Exception, KeyboardInterrupt) as error:
            handle_sampling_error(output_file_path == TMP_FILE_NAME, output_file_path, error)

        else:
            if output_file_path == TMP_FILE_NAME and os.path.exists(output_file_path):
                os.remove(output_file_path)

        return pd.concat(sampled, ignore_index=True) if len(sampled) > 0 else pd.DataFrame()


    def save(self, path):
        self._package_versions = get_package_versions(getattr(self, '_model', None))

        with open(path, 'wb') as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
            throw_version_mismatch_warning(getattr(model, '_package_versions', None))

            return model

            
class CTGANModel(BaseTabularModel):
    """Base class for all the CTGAN models.

    The ``CTGANModel`` class provides a wrapper for all the CTGAN models.
    """

    _MODEL_CLASS = None
    _model_kwargs = None

    _DTYPE_TRANSFORMERS = {
        'O': None
    }

    def _build_model(self):
        return self._MODEL_CLASS(**self._model_kwargs)


    def _sample(self, num_rows, conditions=None):
        if conditions is None:
            return self._model.sample(num_rows)

        raise NotImplementedError(f"{self._MODEL_CLASS} doesn't support conditional sampling.")

    def _set_random_state(self, random_state):
        self._model.set_random_state(random_state)


class QEGAN(CTGANModel):

    _MODEL_CLASS = CTGANSynthesizer

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
                 embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True,
                 rounding='auto', min_value='auto', max_value='auto'):
        super().__init__(
            field_names=field_names,
            primary_key=primary_key,
            field_types=field_types,
            field_transformers=field_transformers,
            anonymize_fields=anonymize_fields,
            constraints=constraints,
            table_metadata=table_metadata,
            rounding=rounding,
            max_value=max_value,
            min_value=min_value
        )

        self._model_kwargs = {
            'embedding_dim': embedding_dim,
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'generator_lr': generator_lr,
            'generator_decay': generator_decay,
            'discriminator_lr': discriminator_lr,
            'discriminator_decay': discriminator_decay,
            'batch_size': batch_size,
            'discriminator_steps': discriminator_steps,
            'log_frequency': log_frequency,
            'verbose': verbose,
            'epochs': epochs,
            'pac': pac,
            'cuda': cuda
        }

        
class Table:
    
    _hyper_transformer = None
    _fakers = None
    _constraint_instances = None
    _fields_metadata = None
    fitted = False

    _ANONYMIZATION_MAPPINGS = dict()
    _TRANSFORMER_TEMPLATES = {
        'integer': rdt.transformers.NumericalTransformer(dtype=int),
        'float': rdt.transformers.NumericalTransformer(dtype=float),
        'categorical': rdt.transformers.CategoricalTransformer,
        'categorical_fuzzy': rdt.transformers.CategoricalTransformer(fuzzy=True),
        'one_hot_encoding': rdt.transformers.OneHotEncodingTransformer,
        'label_encoding': rdt.transformers.LabelEncodingTransformer,
        'boolean': rdt.transformers.BooleanTransformer,
        'datetime': rdt.transformers.DatetimeTransformer(strip_constant=True),
    }
    _DTYPE_TRANSFORMERS = {
        'i': 'integer',
        'f': 'float',
        'O': 'one_hot_encoding',
        'b': 'boolean',
        'M': 'datetime',
    }
    _DTYPES_TO_TYPES = {
        'i': {
            'type': 'numerical',
            'subtype': 'integer',
        },
        'f': {
            'type': 'numerical',
            'subtype': 'float',
        },
        'O': {
            'type': 'categorical',
        },
        'b': {
            'type': 'boolean',
        },
        'M': {
            'type': 'datetime',
        }
    }
    _TYPES_TO_DTYPES = {
        ('categorical', None): 'object',
        ('boolean', None): 'bool',
        ('numerical', None): 'float',
        ('numerical', 'float'): 'float',
        ('numerical', 'integer'): 'int',
        ('datetime', None): 'datetime64',
        ('id', None): 'int',
        ('id', 'integer'): 'int',
        ('id', 'string'): 'str'
    }



    def __init__(self, name=None, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None,
                 dtype_transformers=None, model_kwargs=None, sequence_index=None,
                 entity_columns=None, context_columns=None, rounding=None, min_value=None,
                 max_value=None):
        self.name = name
        self._field_names = field_names
        self._field_types = field_types or {}
        self._field_transformers = field_transformers or {}
        self._anonymize_fields = anonymize_fields or {}
        self._model_kwargs = model_kwargs or {}

        self._primary_key = primary_key
        self._sequence_index = sequence_index
        self._entity_columns = entity_columns or []
        self._context_columns = context_columns or []
        self._constraints = self._prepare_constraints(constraints)
        self._dtype_transformers = self._DTYPE_TRANSFORMERS.copy()
        self._transformer_templates = self._TRANSFORMER_TEMPLATES.copy()
        self._update_transformer_templates(rounding, min_value, max_value)
        if dtype_transformers:
            self._dtype_transformers.update(dtype_transformers)

    def __repr__(self):
        return 'Table(name={}, field_names={})'.format(self.name, self._field_names)

    def get_model_kwargs(self, model_name):
        return copy.deepcopy(self._model_kwargs.get(model_name))

    def set_model_kwargs(self, model_name, model_kwargs):
        """Set the model kwargs used for the indicated model."""
        self._model_kwargs[model_name] = model_kwargs

    def _get_field_dtype(self, field_name, field_metadata):
        field_type = field_metadata['type']
        field_subtype = field_metadata.get('subtype')
        dtype = self._TYPES_TO_DTYPES.get((field_type, field_subtype))
        if not dtype:
            raise MetadataError(
                'Invalid type and subtype combination for field {}: ({}, {})'.format(
                    field_name, field_type, field_subtype)
            )

        return dtype

    def get_fields(self):
        return copy.deepcopy(self._fields_metadata)

    def get_dtypes(self, ids=False):
        dtypes = dict()
        for name, field_meta in self._fields_metadata.items():
            field_type = field_meta['type']

            if ids or (field_type != 'id'):
                dtypes[name] = self._get_field_dtype(name, field_meta)

        return dtypes

    def _build_fields_metadata(self, data):
        fields_metadata = dict()
        for field_name in self._field_names:
            if field_name not in data:
                raise ValueError('Field {} not found in given data'.format(field_name))

            field_meta = self._field_types.get(field_name)
            if field_meta:
                dtype = self._get_field_dtype(field_name, field_meta)
            else:
                dtype = data[field_name].dtype
                field_template = self._DTYPES_TO_TYPES.get(dtype.kind)
                if field_template is None:
                    msg = 'Unsupported dtype {} in column {}'.format(dtype, field_name)
                    raise ValueError(msg)

                field_meta = copy.deepcopy(field_template)

            field_transformer = self._field_transformers.get(field_name)
            if field_transformer:
                field_meta['transformer'] = field_transformer
            else:
                field_meta['transformer'] = self._dtype_transformers.get(np.dtype(dtype).kind)

            anonymize_category = self._anonymize_fields.get(field_name)
            if anonymize_category:
                field_meta['pii'] = True
                field_meta['pii_category'] = anonymize_category

            fields_metadata[field_name] = field_meta

        return fields_metadata

    def _get_transformers(self, dtypes):

        transformers = dict()
        for name, dtype in dtypes.items():
            field_metadata = self._fields_metadata.get(name, {})
            transformer_template = field_metadata.get('transformer')
            if transformer_template is None:
                transformer_template = self._dtype_transformers[np.dtype(dtype).kind]
                if transformer_template is None:
                    # Skip this dtype
                    continue

                field_metadata['transformer'] = transformer_template

            if isinstance(transformer_template, str):
                transformer_template = self._transformer_templates[transformer_template]

            if isinstance(transformer_template, type):
                transformer = transformer_template()
            else:
                transformer = copy.deepcopy(transformer_template)

            LOGGER.debug('Loading transformer %s for field %s',
                         transformer.__class__.__name__, name)
            transformers[name] = transformer

        return transformers


    def reverse_transform(self, data):

        if not self.fitted:
            raise MetadataNotFittedError()

        try:
            reversed_data = self._hyper_transformer.reverse_transform(data)
        except rdt.errors.NotFittedError:
            reversed_data = data

        for constraint in reversed(self._constraints):
            reversed_data = constraint.reverse_transform(reversed_data)

        for name, field_metadata in self._fields_metadata.items():
            field_type = field_metadata['type']
            if field_type == 'id' and name not in reversed_data:
                field_data = self._make_ids(field_metadata, len(reversed_data))
            elif field_metadata.get('pii', False):
                field_data = pd.Series(Table._get_fake_values(field_metadata, len(reversed_data)))
            else:
                field_data = reversed_data[name]

            reversed_data[name] = field_data[field_data.notnull()].astype(self._dtypes[name])

        return reversed_data[self._field_names]

    def filter_valid(self, data):
        for constraint in self._constraints:
            data = constraint.filter_valid(data)

        return data

    def make_ids_unique(self, data):
        for name, field_metadata in self._fields_metadata.items():
            if field_metadata['type'] == 'id' and not data[name].is_unique:
                ids = self._make_ids(field_metadata, len(data))
                ids.index = data.index.copy()
                data[name] = ids

        return data



        
class NonParametricError(Exception):
    """Exception to indicate that a model is not parametric."""


